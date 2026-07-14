#!/usr/bin/env python3
"""Per-disk probe-subspace activation patching at the second SEP token.

Companion to ``probe_subspace_patching.py`` (which patches the 2D distance-matching
probe subspace). Here we patch the subspace spanned by the union of the four
per-disk configuration probes W_0..W_3 (each shape (3, d)) trained in
``representation_analysis.py``. Those probes are not persisted to disk, so we
reproduce them with the exact same procedure (SEP activations of the 81 states,
sklearn multinomial logistic regression, one 3-class probe per disk).

The patch replaces only the ``P_config`` subspace of the recipient SEP activation
with the donor's, keeping the orthogonal complement from the recipient:

    W_all = cat([W_0, W_1, W_2, W_3])          # (12, d)
    basis = right singular vectors of W_all with S > S.max()*1e-6   # (rank, d)
    P_config = basis.T @ basis                 # (d, d), orthogonal projector
    h_patched = P_config @ h_d + (I - P_config) @ h_r

All donor/recipient generation, decoding, and outcome classification are reused
from ``activation_patching.py`` / ``probe_subspace_patching.py`` so the comparison
to the full-residual experiment is exact.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parent.parent))

from toh_transformer import activation_patching as ap
from toh_transformer import probe_subspace_patching as ps
from toh_transformer import representation_analysis as ra
from toh_transformer import utils as utils
from toh_transformer.data import Vocabulary
from toh_transformer.model import ToHTransformer

State = ap.State


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Per-disk probe-subspace activation patching at the second SEP token"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="toh_transformer/checkpoints/n4/best.pt",
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


def build_perdisk_projector(
    model: ToHTransformer,
    states: Sequence[State],
    goal: State,
    vocab: Vocabulary,
    layer: int,
    d_model: int,
    device: torch.device,
) -> Tuple[torch.Tensor, int]:
    """Fit the four per-disk SEP probes and build the projector onto the union of
    their row spaces. Returns (P_config [d,d], rank)."""
    x_sep = ra.extract_sep2_activations_layer(
        model=model, states=states, goal=goal, vocab=vocab, layer=layer, device=device
    )  # (81, d)
    sep_labels = np.array(states, dtype=np.int64)  # (81, 4) peg of each disk
    _, weights = ra.fit_disk_probes_and_weights(x_sep, sep_labels)  # 4 x (3, d)

    W_all_np = np.concatenate(weights, axis=0)  # (12, d)
    if W_all_np.shape != (12, d_model):
        raise ValueError(f"Unexpected stacked probe shape {W_all_np.shape} for d_model={d_model}")
    W_all = torch.tensor(W_all_np, dtype=torch.float32, device=device)

    _, S, Vt = torch.linalg.svd(W_all, full_matrices=False)
    threshold = float(S.max()) * 1e-6
    keep = S > threshold
    basis = Vt[keep]  # (rank, d)
    rank = int(basis.shape[0])
    projector = basis.T @ basis  # (d, d) symmetric orthogonal projector
    return projector, rank


def check_projector(projector: torch.Tensor) -> None:
    """Assert P is symmetric and idempotent (up to float noise)."""
    if not torch.allclose(projector, projector.T, atol=1e-5):
        raise AssertionError("Projector is not symmetric")
    if not torch.allclose(projector @ projector, projector, atol=1e-4):
        raise AssertionError("Projector is not idempotent (P@P != P)")


def print_comparison_table(
    layers: Sequence[int],
    d_model: int,
    full_residual: Dict[int, Dict[str, object]],
    perdisk: Dict[int, Dict[str, object]],
) -> None:
    header = (
        "Layer | Method                       | Dim  | Full  | Partial (K)   | Unchanged | Novel | Disrupted"
    )
    sep = (
        "------+------------------------------+------+-------+---------------+-----------+-------+----------"
    )

    def fmt_row(layer: int, method: str, dim: int, m: Dict[str, object]) -> str:
        full = f"{m['full_transfer_pct']:.2f}%"
        partial = f"{m['partial_pct']:.2f}% ({m['partial_mean_k']:.2f})"
        unch = f"{m['unchanged_pct']:.2f}%"
        novel = f"{m['novel_correct_pct']:.2f}%"
        disr = f"{m['disrupted_pct']:.2f}%"
        return (
            f"{layer:<5d} | {method:<28s} | {dim:<4d} | {full:<5s} | {partial:<13s} | "
            f"{unch:<9s} | {novel:<5s} | {disr}"
        )

    print(header)
    print(sep)
    for layer in layers:
        if layer in full_residual:
            print(fmt_row(layer, "Full residual", d_model, full_residual[layer]))
        if layer in perdisk:
            print(fmt_row(layer, "Per-disk subspace", int(perdisk[layer]["subspace_dim"]), perdisk[layer]))

    print()
    print("Interpretation:")
    print(
        "- If per-disk-subspace patching preserves Full+Partial rates similar to full-residual patching,\n"
        "  the union of the four per-disk subspaces carries the causal effect."
    )
    print(
        "- If Full+Partial rates drop substantially, the causal effect extends beyond the per-disk subspaces."
    )
    print(
        "- Compare to the 2D distance-matching probe patching (probe_subspace_patching.json) to see\n"
        "  which representation localises the effect more tightly."
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

    # Build per-disk projectors once per layer.
    projectors: Dict[int, torch.Tensor] = {}
    ranks: Dict[int, int] = {}
    for layer in layers:
        projector, rank = build_perdisk_projector(
            model=model, states=all_states, goal=goal, vocab=vocab,
            layer=layer, d_model=d_model, device=device,
        )
        if not (3 <= rank <= 12):
            raise AssertionError(f"Layer {layer}: per-disk subspace rank {rank} outside [3, 12]")
        check_projector(projector)
        projectors[layer] = projector
        ranks[layer] = rank
        print(f"[INFO] Layer {layer}: per-disk subspace rank = {rank} (projector symmetric + idempotent OK)")

    # Sanity check: h_donor == h_recipient must be a no-op (Unchanged).
    unchanged, checked = ps.sanity_check_noop_patch(
        model=model, vocab=vocab, goal=goal, sep2_idx=sep2_idx, layer=layers[0],
        projector=projectors[layers[0]], valid_states=valid_states, clean_by_state=clean_by_state,
        cached_sep_by_layer=cached_sep_by_layer, state_to_all_index=state_to_all_index, device=device,
    )
    print(f"[SANITY] no-op patch (h_d==h_r) unchanged {unchanged}/{checked} at layer {layers[0]}")
    if unchanged != checked:
        raise AssertionError("Sanity check failed: no-op per-disk patch changed the decode")

    perdisk_summ: Dict[int, Dict[str, object]] = {}
    for layer in layers:
        print(f"[INFO] Running per-disk-subspace patch sweep for layer {layer}...")
        rows = ps.run_layer_probe_patch(
            model=model, vocab=vocab, goal=goal, sep2_idx=sep2_idx, layer=layer,
            projector=projectors[layer], pairs=all_valid_pairs, clean_by_state=clean_by_state,
            cached_sep_by_layer=cached_sep_by_layer, state_to_all_index=state_to_all_index, device=device,
        )
        summ = ps.summarize_to_schema(rows)
        summ["subspace_dim"] = ranks[layer]
        perdisk_summ[layer] = summ

    full_residual_summ = ps.load_full_residual_by_layer(Path(args.full_residual_results), layers)

    result = {
        "config": {
            "checkpoint": str(checkpoint_path),
            "probe_checkpoint": "toh_transformer/representation_analysis.py::fit_disk_probes_and_weights (refit from SEP activations)",
            "layers": list(layers),
            "n_donor_recipient_pairs": len(all_valid_pairs),
        },
        "results_by_layer": {
            str(layer): {
                "n_pairs": perdisk_summ[layer]["n_pairs"],
                "subspace_dim": perdisk_summ[layer]["subspace_dim"],
                "full_transfer_pct": perdisk_summ[layer]["full_transfer_pct"],
                "partial_pct": perdisk_summ[layer]["partial_pct"],
                "partial_mean_k": perdisk_summ[layer]["partial_mean_k"],
                "unchanged_pct": perdisk_summ[layer]["unchanged_pct"],
                "novel_correct_pct": perdisk_summ[layer]["novel_correct_pct"],
                "disrupted_pct": perdisk_summ[layer]["disrupted_pct"],
            }
            for layer in layers
        },
    }
    out_path = output_dir / "perdisk_subspace_patching.json"
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"[INFO] Wrote {out_path}")

    print()
    print_comparison_table(layers, d_model, full_residual_summ, perdisk_summ)


if __name__ == "__main__":
    main()
