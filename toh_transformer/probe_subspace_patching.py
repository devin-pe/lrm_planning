#!/usr/bin/env python3
"""Probe-subspace activation patching at the second SEP token.

This is a more surgical variant of the full-residual activation patching in
``activation_patching.py``. Instead of overwriting the entire recipient residual
at the second SEP token with the donor's activation, we replace only the 2D
subspace read by the trained distance-matching probe (probe.py) and keep all
other d - 2 dimensions from the recipient.

Given the probe weight matrix W (shape 2 x d), an orthonormal basis for its row
space defines an orthogonal projector P onto the probe subspace. The patch is

    h_patched = P @ h_d + (I - P) @ h_r
              = h_r + P @ (h_d - h_r)

where h_d / h_r are the donor / recipient SEP activations. All donor/recipient
pair generation, decoding, and outcome classification are reused from
``activation_patching.py`` so the comparison to the full-residual experiment is
exact.
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Probe-subspace (2D) activation patching at the second SEP token")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="toh_transformer/checkpoints/n4/best.pt",
        help="Model checkpoint (epoch-50 weights; identical to epoch_0050.pt).",
    )
    parser.add_argument(
        "--probe_dir",
        type=str,
        default="toh_transformer/probe/output_0050",
        help="Directory holding sep_probe_layer_{NN}.pt from probe.py (epoch 50).",
    )
    parser.add_argument("--n_disks", type=int, default=4)
    parser.add_argument(
        "--output_dir",
        type=str,
        default="toh_transformer/patching_output",
    )
    parser.add_argument("--layers", type=str, default="4,5,6")
    parser.add_argument(
        "--full_residual_results",
        type=str,
        default="toh_transformer/activation_patching_output/activation_patching_results.json",
        help="Existing full-residual raw results, loaded for the exact comparison table.",
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


def load_probe_projector(probe_dir: Path, layer: int, d_model: int, device: torch.device) -> Tuple[torch.Tensor, int]:
    """Load the 2xd probe weight matrix for a layer and build the orthogonal
    projector onto its row space. Returns (projector [d,d], subspace_dim)."""
    probe_path = probe_dir / f"sep_probe_layer_{layer:02d}.pt"
    if not probe_path.exists():
        raise FileNotFoundError(f"Probe weights not found: {probe_path}")

    state_dict = torch.load(probe_path, map_location=device)
    W = state_dict["weight"].to(device=device, dtype=torch.float32)  # (2, d)
    if W.dim() != 2 or W.shape[1] != d_model:
        raise ValueError(f"Unexpected probe weight shape {tuple(W.shape)} for d_model={d_model}")

    # Orthonormal basis for the probe's row space (torch.linalg keeps it on GPU).
    _, S, Vt = torch.linalg.svd(W, full_matrices=False)
    subspace_dim = int((S > 1e-6 * float(S[0])).sum().item())
    basis = Vt  # (2, d) orthonormal rows spanning the probe subspace
    projector = basis.T @ basis  # (d, d) symmetric orthogonal projector
    return projector, subspace_dim


@torch.no_grad()
def forward_with_sep_probe_patch(
    model: ToHTransformer,
    input_ids: torch.Tensor,
    target_layer: int,
    sep_index: int,
    donor_sep_activation: torch.Tensor,
    projector: torch.Tensor,
) -> torch.Tensor:
    """Forward pass patching only the probe subspace at one position.

    Mirrors ``activation_patching.forward_with_sep_patch`` but replaces the
    residual splice with h_patched = h_r + P @ (h_d - h_r), where h_r is the
    recipient's in-flight activation at ``sep_index`` and h_d the donor's.
    """
    bsz, seq_len = input_ids.shape
    if seq_len > model.max_seq_len:
        raise ValueError(f"Sequence length {seq_len} exceeds max_seq_len={model.max_seq_len}")
    if target_layer < 1 or target_layer > model.n_layers:
        raise ValueError(f"target_layer must be in [1, {model.n_layers}], got {target_layer}")

    if donor_sep_activation.dim() == 1:
        donor_act = donor_sep_activation.unsqueeze(0).expand(bsz, -1)
    elif donor_sep_activation.dim() == 2 and donor_sep_activation.size(0) == bsz:
        donor_act = donor_sep_activation
    else:
        raise ValueError("donor_sep_activation must be shape [d_model] or [batch, d_model]")

    positions = torch.arange(0, seq_len, device=input_ids.device, dtype=torch.long)
    x = model.token_emb(input_ids) + model.pos_emb(positions).unsqueeze(0)
    x = model.emb_dropout(x)

    for layer_idx, block in enumerate(model.blocks, start=1):
        x = block(x)
        if layer_idx == target_layer:
            x = x.clone()
            h_r = x[:, sep_index, :]  # (bsz, d) recipient activation in-flight
            # P is symmetric, so (h_d - h_r) @ P == P @ (h_d - h_r) row-wise.
            h_patched = h_r + (donor_act - h_r) @ projector
            x[:, sep_index, :] = h_patched

    x = model.ln_f(x)
    logits = model.lm_head(x)
    return logits


@torch.no_grad()
def greedy_decode_ids_with_probe_patch(
    model: ToHTransformer,
    context_ids: Sequence[int],
    target_layer: int,
    sep_index: int,
    donor_sep_activation: torch.Tensor,
    projector: torch.Tensor,
    eos_id: int,
    device: torch.device,
) -> Tuple[List[int], bool]:
    """Autoregressive greedy decode, reapplying the probe-subspace patch each step."""
    seq = torch.tensor(context_ids, dtype=torch.long, device=device).unsqueeze(0)
    generated: List[int] = []
    donor_act = donor_sep_activation.to(device=device)

    while seq.size(1) < model.max_seq_len:
        logits = forward_with_sep_probe_patch(
            model=model,
            input_ids=seq,
            target_layer=target_layer,
            sep_index=sep_index,
            donor_sep_activation=donor_act,
            projector=projector,
        )
        next_id = int(torch.argmax(logits[:, -1, :], dim=-1).item())
        generated.append(next_id)
        seq = torch.cat([seq, torch.tensor([[next_id]], dtype=torch.long, device=device)], dim=1)
        if next_id == eos_id:
            return generated, True

    return generated, False


def run_layer_probe_patch(
    model: ToHTransformer,
    vocab: Vocabulary,
    goal: State,
    sep2_idx: int,
    layer: int,
    projector: torch.Tensor,
    pairs: Sequence[Pair],
    clean_by_state: Dict[State, Dict[str, object]],
    cached_sep_by_layer: Dict[int, torch.Tensor],
    state_to_all_index: Dict[State, int],
    device: torch.device,
) -> List[Dict[str, object]]:
    """Probe-subspace analog of ap.run_layer_on_pairs (reuses its classifier/parser)."""
    layer_cache = cached_sep_by_layer[layer]
    rows: List[Dict[str, object]] = []

    for donor, recipient in pairs:
        donor_vec = layer_cache[state_to_all_index[donor]]
        recipient_context = utils.build_context_ids(recipient, goal, vocab)

        patched_ids, patched_eos = greedy_decode_ids_with_probe_patch(
            model=model,
            context_ids=recipient_context,
            target_layer=layer,
            sep_index=sep2_idx,
            donor_sep_activation=donor_vec,
            projector=projector,
            eos_id=vocab.eos_id,
            device=device,
        )

        parsed = ap.parse_generation(
            start=recipient,
            goal=goal,
            generated_ids=patched_ids,
            eos_seen=patched_eos,
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

        rows.append(
            {
                "donor": list(donor),
                "recipient": list(recipient),
                "layer": layer,
                "category": category,
                "partial_k": partial_k,
                "patched_reaches_goal": bool(parsed["reaches_goal"]),
            }
        )

    return rows


def sanity_check_noop_patch(
    model: ToHTransformer,
    vocab: Vocabulary,
    goal: State,
    sep2_idx: int,
    layer: int,
    projector: torch.Tensor,
    valid_states: Sequence[State],
    clean_by_state: Dict[State, Dict[str, object]],
    cached_sep_by_layer: Dict[int, torch.Tensor],
    state_to_all_index: Dict[State, int],
    device: torch.device,
    n_sample: int = 20,
) -> Tuple[int, int]:
    """Inject each recipient's OWN SEP activation (h_d == h_r) -> patch is a no-op,
    so decoded moves must equal the clean recipient decode (Unchanged). Returns
    (n_unchanged, n_checked)."""
    layer_cache = cached_sep_by_layer[layer]
    checked = 0
    unchanged = 0
    for recipient in list(valid_states)[:n_sample]:
        own_vec = layer_cache[state_to_all_index[recipient]]
        context = utils.build_context_ids(recipient, goal, vocab)
        patched_ids, patched_eos = greedy_decode_ids_with_probe_patch(
            model=model,
            context_ids=context,
            target_layer=layer,
            sep_index=sep2_idx,
            donor_sep_activation=own_vec,
            projector=projector,
            eos_id=vocab.eos_id,
            device=device,
        )
        parsed = ap.parse_generation(
            start=recipient, goal=goal, generated_ids=patched_ids, eos_seen=patched_eos, vocab=vocab
        )
        checked += 1
        if list(parsed["moves"]) == list(clean_by_state[recipient]["moves"]):
            unchanged += 1
    return unchanged, checked


def summarize_to_schema(rows: Sequence[Dict[str, object]]) -> Dict[str, object]:
    s = ap.summarize_layer(rows)
    rates = s["rates"]
    return {
        "n_pairs": int(s["n_pairs"]),
        "full_transfer_pct": 100.0 * float(rates[ap.CATEGORY_FULL_TRANSFER]),
        "partial_pct": 100.0 * float(rates[ap.CATEGORY_PARTIAL_TRANSFER]),
        "partial_mean_k": float(s["partial_mean_k"]),
        "unchanged_pct": 100.0 * float(rates[ap.CATEGORY_RECIPIENT_UNCHANGED]),
        "novel_correct_pct": 100.0 * float(rates[ap.CATEGORY_NOVEL_CORRECT]),
        "disrupted_pct": 100.0 * float(rates[ap.CATEGORY_NOVEL_INCORRECT]),
    }


def load_full_residual_by_layer(results_path: Path, layers: Sequence[int]) -> Dict[int, Dict[str, object]]:
    """Recompute per-layer rates from the existing full-residual results file so
    the comparison row is exact."""
    rows = json.loads(results_path.read_text(encoding="utf-8"))
    out: Dict[int, Dict[str, object]] = {}
    for layer in layers:
        layer_rows = [r for r in rows if int(r["layer"]) == layer]
        if layer_rows:
            out[layer] = summarize_to_schema(layer_rows)
    return out


def print_comparison_table(
    layers: Sequence[int],
    full_residual: Dict[int, Dict[str, object]],
    probe_subspace: Dict[int, Dict[str, object]],
) -> None:
    header = (
        "Layer | Method              | Full Transfer | Partial (K)   | Unchanged | Novel  | Disrupted"
    )
    sep = "------+---------------------+---------------+---------------+-----------+--------+----------"

    def fmt_row(layer: int, method: str, m: Dict[str, object]) -> str:
        full = f"{m['full_transfer_pct']:.2f}%"
        partial = f"{m['partial_pct']:.2f}% ({m['partial_mean_k']:.2f})"
        unch = f"{m['unchanged_pct']:.2f}%"
        novel = f"{m['novel_correct_pct']:.2f}%"
        disr = f"{m['disrupted_pct']:.2f}%"
        return (
            f"{layer:<5d} | {method:<19s} | {full:<13s} | {partial:<13s} | "
            f"{unch:<9s} | {novel:<6s} | {disr}"
        )

    print(header)
    print(sep)
    for layer in layers:
        if layer in full_residual:
            print(fmt_row(layer, "Full residual", full_residual[layer]))
        if layer in probe_subspace:
            print(fmt_row(layer, "Probe subspace (2d)", probe_subspace[layer]))

    print()
    print("Interpretation:")
    print(
        "- If probe-subspace patching preserves Full+Partial rates similar to full-residual patching,\n"
        "  the 2D world-model subspace carries the causal effect."
    )
    print(
        "- If Full+Partial rates drop substantially under probe-subspace patching,\n"
        "  the full-residual causal effect was distributed across other dimensions beyond the probe subspace."
    )


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

    checkpoint_path = utils.resolve_checkpoint_path(args.checkpoint, args.n_disks)
    model = ap.load_model(checkpoint_path, args.n_disks, device)
    d_model = model.token_emb.embedding_dim

    goal: State = tuple(2 for _ in range(args.n_disks))
    all_states = ap.enumerate_states(args.n_disks)
    state_to_all_index = {s: i for i, s in enumerate(all_states)}

    sep2_idx = 2 * args.n_disks + 2
    if sep2_idx != 10:
        raise AssertionError(f"Expected second SEP index 10 for n=4, got {sep2_idx}")

    print("[INFO] Caching clean second-SEP activations for all starts and layers...")
    cached_sep_by_layer = ap.extract_sep2_activations(
        model=model, starts=all_states, goal=goal, vocab=vocab, device=device, batch_size=128
    )

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

    # Build projectors once per layer.
    projectors: Dict[int, torch.Tensor] = {}
    subspace_dims = set()
    for layer in layers:
        projector, dim = load_probe_projector(Path(args.probe_dir), layer, d_model, device)
        projectors[layer] = projector
        subspace_dims.add(dim)
        print(f"[INFO] Layer {layer}: probe subspace dim = {dim}")
    if subspace_dims != {2}:
        raise AssertionError(f"Expected all probe subspaces to be 2D, got dims {sorted(subspace_dims)}")

    # Sanity check: h_donor == h_recipient must be a no-op (Unchanged).
    unchanged, checked = sanity_check_noop_patch(
        model=model, vocab=vocab, goal=goal, sep2_idx=sep2_idx, layer=layers[0],
        projector=projectors[layers[0]], valid_states=valid_states, clean_by_state=clean_by_state,
        cached_sep_by_layer=cached_sep_by_layer, state_to_all_index=state_to_all_index, device=device,
    )
    print(f"[SANITY] no-op patch (h_d==h_r) unchanged {unchanged}/{checked} at layer {layers[0]}")
    if unchanged != checked:
        raise AssertionError("Sanity check failed: no-op probe patch changed the decode")

    probe_subspace_summ: Dict[int, Dict[str, object]] = {}
    for layer in layers:
        print(f"[INFO] Running probe-subspace patch sweep for layer {layer}...")
        rows = run_layer_probe_patch(
            model=model, vocab=vocab, goal=goal, sep2_idx=sep2_idx, layer=layer,
            projector=projectors[layer], pairs=all_valid_pairs, clean_by_state=clean_by_state,
            cached_sep_by_layer=cached_sep_by_layer, state_to_all_index=state_to_all_index, device=device,
        )
        probe_subspace_summ[layer] = summarize_to_schema(rows)

    full_residual_summ = load_full_residual_by_layer(Path(args.full_residual_results), layers)

    result = {
        "config": {
            "checkpoint": str(checkpoint_path),
            "probe_checkpoint": str(Path(args.probe_dir)),
            "layers": list(layers),
            "n_donor_recipient_pairs": len(all_valid_pairs),
        },
        "results_by_layer": {str(layer): probe_subspace_summ[layer] for layer in layers},
        "probe_subspace_dim": 2,
    }
    out_path = output_dir / "probe_subspace_patching.json"
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"[INFO] Wrote {out_path}")

    print()
    print_comparison_table(layers, full_residual_summ, probe_subspace_summ)


if __name__ == "__main__":
    main()
