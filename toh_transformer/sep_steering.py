#!/usr/bin/env python3
"""Soft activation steering at the second SEP token.

Where ``activation_patching.py`` fully replaces the recipient's SEP residual with
the donor's activation (hard patching) and ``probe_subspace_patching.py`` replaces
only the 2D probe subspace, this script *steers*: it adds a scaled, unit-normalised
donor direction to the recipient's in-flight SEP activation, rather than replacing
anything. Sweeping the injection strength ``alpha`` shows whether the causal effect
is graded or threshold-driven.

For a donor/recipient pair at a target layer:

    h_donor      = cached clean SEP activation of the donor
    h_recipient  = recipient SEP activation in-flight during the forward pass
    h_bar        = mean SEP activation over all 81 configurations (per layer)
    direction    = (h_donor - h_bar) / ||h_donor - h_bar||_2      # unit norm
    h_tilde      = h_recipient + alpha * direction                # steered

The direction depends only on the donor (and the per-layer mean), so
``alpha * direction`` is a fixed additive vector reapplied at the SEP position on
every decode step -- exactly mirroring the hard-patching forward pass, but adding
instead of replacing.

All donor/recipient pair generation, decoding, outcome classification and
summarisation are reused from ``activation_patching.py`` so the comparison to the
hard-patching experiment is exact.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import torch

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parent.parent))

from toh_transformer import activation_patching as ap
from toh_transformer import utils as utils
from toh_transformer.data import Vocabulary
from toh_transformer.model import ToHTransformer

State = ap.State
Pair = ap.Pair

DEFAULT_ALPHAS = [0.0, 0.5, 1.0, 2.0, 5.0, 10.0]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Soft activation steering at the second SEP token")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="toh_transformer/checkpoints/n4/best.pt",
        help="Model checkpoint (epoch-50 weights; identical to epoch_0050.pt).",
    )
    parser.add_argument("--n_disks", type=int, default=4)
    parser.add_argument(
        "--output_dir",
        type=str,
        default="toh_transformer/patching_output",
    )
    parser.add_argument("--layers", type=str, default="4,5,6")
    parser.add_argument(
        "--alphas",
        type=str,
        default=",".join(str(a) for a in DEFAULT_ALPHAS),
        help="Comma-separated steering strengths.",
    )
    parser.add_argument(
        "--full_residual_results",
        type=str,
        default="toh_transformer/activation_patching_output/activation_patching_results.json",
        help="Existing hard-patching raw results, loaded for the exact comparison table.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--max_pairs",
        type=int,
        default=0,
        help="If >0, cap donor->recipient pairs per layer (smoke testing only). 0 = all pairs.",
    )
    return parser.parse_args()


def parse_alpha_list(spec: str) -> List[float]:
    values: List[float] = []
    for part in spec.split(","):
        token = part.strip()
        if not token:
            continue
        values.append(float(token))
    if not values:
        raise ValueError("Alpha list is empty")
    # Preserve order while dropping duplicates.
    seen = set()
    out: List[float] = []
    for v in values:
        if v not in seen:
            seen.add(v)
            out.append(v)
    return out


@torch.no_grad()
def forward_with_sep_steer(
    model: ToHTransformer,
    input_ids: torch.Tensor,
    target_layer: int,
    sep_index: int,
    steer_vec: torch.Tensor,
) -> torch.Tensor:
    """Forward pass adding ``steer_vec`` to the SEP residual at one position.

    Mirrors ``activation_patching.forward_with_sep_patch`` but adds a fixed vector
    (``alpha * direction``) to the recipient's in-flight activation instead of
    overwriting it. ``steer_vec`` can be shape [d_model] or [batch, d_model].
    """
    bsz, seq_len = input_ids.shape
    if seq_len > model.max_seq_len:
        raise ValueError(f"Sequence length {seq_len} exceeds max_seq_len={model.max_seq_len}")
    if target_layer < 1 or target_layer > model.n_layers:
        raise ValueError(f"target_layer must be in [1, {model.n_layers}], got {target_layer}")

    if steer_vec.dim() == 1:
        add = steer_vec.unsqueeze(0).expand(bsz, -1)
    elif steer_vec.dim() == 2 and steer_vec.size(0) == bsz:
        add = steer_vec
    else:
        raise ValueError("steer_vec must be shape [d_model] or [batch, d_model]")

    positions = torch.arange(0, seq_len, device=input_ids.device, dtype=torch.long)
    x = model.token_emb(input_ids) + model.pos_emb(positions).unsqueeze(0)
    x = model.emb_dropout(x)

    for layer_idx, block in enumerate(model.blocks, start=1):
        x = block(x)
        if layer_idx == target_layer:
            x = x.clone()
            x[:, sep_index, :] = x[:, sep_index, :] + add

    x = model.ln_f(x)
    logits = model.lm_head(x)
    return logits


@torch.no_grad()
def greedy_decode_ids_with_steer(
    model: ToHTransformer,
    context_ids: Sequence[int],
    target_layer: int,
    sep_index: int,
    steer_vec: torch.Tensor,
    eos_id: int,
    device: torch.device,
) -> Tuple[List[int], bool]:
    """Autoregressive greedy decode, reapplying the SEP steering vector each step."""
    seq = torch.tensor(context_ids, dtype=torch.long, device=device).unsqueeze(0)
    generated: List[int] = []
    add = steer_vec.to(device=device)

    while seq.size(1) < model.max_seq_len:
        logits = forward_with_sep_steer(
            model=model,
            input_ids=seq,
            target_layer=target_layer,
            sep_index=sep_index,
            steer_vec=add,
        )
        next_id = int(torch.argmax(logits[:, -1, :], dim=-1).item())
        generated.append(next_id)
        seq = torch.cat([seq, torch.tensor([[next_id]], dtype=torch.long, device=device)], dim=1)
        if next_id == eos_id:
            return generated, True

    return generated, False


def compute_directions(layer_cache: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Given cached SEP activations (n_states, d), return (h_bar, directions).

    directions[i] = (h_i - h_bar) / ||h_i - h_bar|| is the unit steering direction
    for donor state i. h_bar is the mean over all states.
    """
    h_bar = layer_cache.mean(dim=0)  # (d,)
    diff = layer_cache - h_bar.unsqueeze(0)  # (n, d)
    norms = diff.norm(dim=1, keepdim=True)  # (n, 1)
    # Guard against a zero-norm donor (a state exactly at the mean); leave it zero.
    safe = torch.where(norms > 0, norms, torch.ones_like(norms))
    directions = diff / safe
    return h_bar, directions


def run_layer_steer_sweep(
    model: ToHTransformer,
    vocab: Vocabulary,
    goal: State,
    sep2_idx: int,
    layer: int,
    alphas: Sequence[float],
    directions: torch.Tensor,
    pairs: Sequence[Pair],
    clean_by_state: Dict[State, Dict[str, object]],
    state_to_all_index: Dict[State, int],
    device: torch.device,
) -> Dict[float, Dict[str, object]]:
    """Run the full donor->recipient sweep for every alpha at one layer.

    Returns {alpha: schema_summary}. Reuses ap.classify_patched_output /
    ap.parse_generation / ap.summarize_layer for exact parity with hard patching.
    """
    summaries: Dict[float, Dict[str, object]] = {}

    for alpha in alphas:
        rows: List[Dict[str, object]] = []
        for donor, recipient in pairs:
            direction = directions[state_to_all_index[donor]]
            steer_vec = alpha * direction

            recipient_context = utils.build_context_ids(recipient, goal, vocab)
            gen_ids, eos_seen = greedy_decode_ids_with_steer(
                model=model,
                context_ids=recipient_context,
                target_layer=layer,
                sep_index=sep2_idx,
                steer_vec=steer_vec,
                eos_id=vocab.eos_id,
                device=device,
            )

            parsed = ap.parse_generation(
                start=recipient,
                goal=goal,
                generated_ids=gen_ids,
                eos_seen=eos_seen,
                vocab=vocab,
            )

            donor_moves = list(clean_by_state[donor]["moves"])
            recipient_moves = list(clean_by_state[recipient]["moves"])
            patched_moves = list(parsed["moves"])

            category, partial_k = ap.classify_patched_output(
                patched_moves=patched_moves,
                donor_moves=donor_moves,
                recipient_moves=recipient_moves,
                patched_is_correct=bool(parsed["reaches_goal"]),
            )
            rows.append({"category": category, "partial_k": partial_k})

        summaries[alpha] = summarize_to_schema(ap.summarize_layer(rows))
        m = summaries[alpha]
        print(
            f"[INFO]   layer {layer} alpha={alpha:<5g} | "
            f"Full {m['full_pct']:.2f}% | Partial {m['partial_pct']:.2f}% (K={m['partial_mean_k']:.2f}) | "
            f"Unchanged {m['unchanged_pct']:.2f}% | Novel {m['novel_correct_pct']:.2f}% | "
            f"Disrupted {m['disrupted_pct']:.2f}%"
        )

    return summaries


def summarize_to_schema(s: Dict[str, object]) -> Dict[str, object]:
    rates = s["rates"]
    return {
        "full_pct": 100.0 * float(rates[ap.CATEGORY_FULL_TRANSFER]),
        "partial_pct": 100.0 * float(rates[ap.CATEGORY_PARTIAL_TRANSFER]),
        "partial_mean_k": float(s["partial_mean_k"]),
        "unchanged_pct": 100.0 * float(rates[ap.CATEGORY_RECIPIENT_UNCHANGED]),
        "novel_correct_pct": 100.0 * float(rates[ap.CATEGORY_NOVEL_CORRECT]),
        "disrupted_pct": 100.0 * float(rates[ap.CATEGORY_NOVEL_INCORRECT]),
    }


def load_hard_patching_by_layer(results_path: Path, layers: Sequence[int]) -> Dict[int, Dict[str, object]]:
    """Recompute per-layer hard-patching rates from the existing full-residual
    results file so the comparison row is exact."""
    rows = json.loads(results_path.read_text(encoding="utf-8"))
    out: Dict[int, Dict[str, object]] = {}
    for layer in layers:
        layer_rows = [r for r in rows if int(r["layer"]) == layer]
        if layer_rows:
            out[layer] = summarize_to_schema(ap.summarize_layer(layer_rows))
    return out


def print_alpha_table(layer: int, alphas: Sequence[float], summaries: Dict[float, Dict[str, object]]) -> None:
    print(f"Layer {layer} -- SEP steering alpha sweep")
    print("alpha | Full  | Partial (K)   | Unchanged | Novel | Disrupted")
    print("------+-------+---------------+-----------+-------+----------")
    for alpha in alphas:
        m = summaries[alpha]
        partial = f"{m['partial_pct']:.2f}% ({m['partial_mean_k']:.2f})"
        print(
            f"{alpha:<5g} | "
            f"{m['full_pct']:.2f}% | "
            f"{partial:<13s} | "
            f"{m['unchanged_pct']:.2f}%    | "
            f"{m['novel_correct_pct']:.2f}% | "
            f"{m['disrupted_pct']:.2f}%"
        )
    print()


def print_comparison_table(
    layer: int,
    steer_summary: Dict[str, object],
    steer_alpha: float,
    hard: Dict[str, object],
) -> None:
    print(f"Layer {layer} -- Steering vs hard patching (full residual)")
    print("Method              | Full  | Partial (K)   | Unchanged | Novel | Disrupted")
    print("--------------------+-------+---------------+-----------+-------+----------")

    def fmt(method: str, m: Dict[str, object]) -> str:
        partial = f"{m['partial_pct']:.2f}% ({m['partial_mean_k']:.2f})"
        return (
            f"{method:<19s} | "
            f"{m['full_pct']:.2f}% | "
            f"{partial:<13s} | "
            f"{m['unchanged_pct']:.2f}%    | "
            f"{m['novel_correct_pct']:.2f}% | "
            f"{m['disrupted_pct']:.2f}%"
        )

    print(fmt(f"Steering alpha={steer_alpha:g}", steer_summary))
    print(fmt("Hard patching", hard))
    print()


def print_interpretation() -> None:
    print("Interpretation:")
    print(
        "- Monotonic increase in Full+Partial with alpha: intervention effect is graded.\n"
        "  The stronger the steering, the more the recipient's solution shifts toward the donor."
    )
    print(
        "- Saturation at moderate alpha: the donor's representation completely dominates\n"
        "  the recipient's beyond a threshold strength."
    )
    print(
        "- Non-monotonic (peak then decline): moderate steering helps but large steering\n"
        "  pushes the residual stream out of the manifold where the model computes correctly."
    )
    print(
        "- Steering alpha=1.0 typically comparable to hard patching in magnitude, since\n"
        "  the direction is unit-normalised and alpha=1 adds a vector of similar magnitude\n"
        "  to the difference (h_donor - h_bar) itself."
    )
    print()


def main() -> None:
    args = parse_args()
    if args.n_disks != 4:
        raise ValueError("This experiment is defined for n_disks=4")

    utils.set_seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    vocab = Vocabulary()
    utils.confirm_tokenizer_mapping(vocab)

    layers = ap.parse_layer_list(args.layers)
    alphas = parse_alpha_list(args.alphas)

    checkpoint_path = utils.resolve_checkpoint_path(args.checkpoint, args.n_disks)
    model = ap.load_model(checkpoint_path, args.n_disks, device)

    goal: State = tuple(2 for _ in range(args.n_disks))
    all_states = ap.enumerate_states(args.n_disks)
    state_to_all_index = {s: i for i, s in enumerate(all_states)}

    sep2_idx = 2 * args.n_disks + 2
    if sep2_idx != 10:
        raise AssertionError(f"Expected second SEP index 10 for n=4, got {sep2_idx}")

    # --- Precomputation: cache SEP activations, build per-layer directions. ---
    print("[INFO] Caching clean second-SEP activations for all starts and layers...")
    cached_sep_by_layer = ap.extract_sep2_activations(
        model=model, starts=all_states, goal=goal, vocab=vocab, device=device, batch_size=128
    )

    directions_by_layer: Dict[int, torch.Tensor] = {}
    for layer in layers:
        layer_cache = cached_sep_by_layer[layer].to(device=device, dtype=torch.float32)
        if layer_cache.shape[0] != len(all_states):
            raise AssertionError(
                f"Expected {len(all_states)} cached SEP activations at layer {layer}, "
                f"got {layer_cache.shape[0]}"
            )
        _, directions = compute_directions(layer_cache)
        directions_by_layer[layer] = directions

        # Sanity: every steering direction must be unit norm (excluding any donor
        # that sits exactly at the mean, which stays a zero vector).
        norms = directions.norm(dim=1)
        nonzero = norms[norms > 0]
        max_dev = float((nonzero - 1.0).abs().max().item()) if nonzero.numel() else 0.0
        if max_dev > 1e-4:
            raise AssertionError(
                f"Layer {layer}: steering directions not unit norm (max deviation {max_dev:.2e})"
            )
        print(f"[SANITY] Layer {layer}: {nonzero.numel()}/{norms.numel()} directions unit norm "
              f"(max |norm-1| = {max_dev:.2e})")

    # --- Clean greedy decode for all 81 starts (donor/recipient move sequences). ---
    print("[INFO] Running clean greedy decode for all 81 starts...")
    clean_by_state: Dict[State, Dict[str, object]] = {}
    valid_states: List[State] = []
    for st in all_states:
        context = utils.build_context_ids(st, goal, vocab)
        gen_ids, eos_seen = utils.greedy_decode_ids(
            model=model, context_ids=context, eos_id=vocab.eos_id, device=device
        )
        parsed = ap.parse_generation(
            start=st, goal=goal, generated_ids=gen_ids, eos_seen=eos_seen, vocab=vocab
        )
        clean_by_state[st] = parsed
        if bool(parsed["reaches_goal"]):
            valid_states.append(st)

    print(f"[INFO] Correct clean solutions: {len(valid_states)} / {len(all_states)}")
    all_valid_pairs = [(a, b) for a in valid_states for b in valid_states if a != b]
    if not all_valid_pairs:
        raise RuntimeError("No valid donor/recipient pairs available")
    if args.max_pairs and args.max_pairs > 0:
        all_valid_pairs = all_valid_pairs[: args.max_pairs]
        print(f"[WARN] SMOKE TEST: capping to {len(all_valid_pairs)} pairs (--max_pairs)")
    print(f"[INFO] Ordered donor->recipient pairs per layer: {len(all_valid_pairs)}")
    print(f"[INFO] Alpha sweep: {alphas}")

    # --- Steering sweep. ---
    summaries_by_layer: Dict[int, Dict[float, Dict[str, object]]] = {}
    for layer in layers:
        print(f"[INFO] Running steering alpha sweep for layer {layer}...")
        summaries_by_layer[layer] = run_layer_steer_sweep(
            model=model, vocab=vocab, goal=goal, sep2_idx=sep2_idx, layer=layer,
            alphas=alphas, directions=directions_by_layer[layer], pairs=all_valid_pairs,
            clean_by_state=clean_by_state, state_to_all_index=state_to_all_index, device=device,
        )

    # Sanity: alpha=0 must reproduce recipient behaviour (100% Unchanged).
    if 0.0 in alphas:
        for layer in layers:
            unch = float(summaries_by_layer[layer][0.0]["unchanged_pct"])
            if abs(unch - 100.0) > 1e-6:
                raise AssertionError(
                    f"Layer {layer}: alpha=0 gave {unch:.4f}% Unchanged, expected 100% (no intervention)"
                )
        print("[SANITY] alpha=0 gave 100% Unchanged at every layer.")

    hard_by_layer = load_hard_patching_by_layer(Path(args.full_residual_results), layers)

    # --- JSON output (schema per task spec). ---
    result = {
        "config": {
            "checkpoint": str(checkpoint_path),
            "layers": list(layers),
            "alpha_values": list(alphas),
            "reference_goal": list(goal),
            "n_donor_recipient_pairs": len(all_valid_pairs),
        },
        "results": {
            f"layer_{layer}": {
                f"alpha_{alpha:g}": summaries_by_layer[layer][alpha] for alpha in alphas
            }
            for layer in layers
        },
    }
    out_path = output_dir / "sep_steering_alpha_sweep.json"
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"[INFO] Wrote {out_path}")
    print()

    # --- Tables. ---
    # Comparison uses the largest alpha (or 5.0 if present) as the representative
    # strong-steering row, matching the task's example.
    compare_alpha = 5.0 if 5.0 in alphas else alphas[-1]
    for layer in layers:
        print_alpha_table(layer, alphas, summaries_by_layer[layer])
        if layer in hard_by_layer:
            print_comparison_table(
                layer,
                summaries_by_layer[layer][compare_alpha],
                compare_alpha,
                hard_by_layer[layer],
            )

    print_interpretation()
    print("[INFO] Done.")


if __name__ == "__main__":
    main()
