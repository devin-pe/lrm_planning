#!/usr/bin/env python3
"""Activation patching using the MLP-encoder's readable subspace.

Loads a trained joint-MLP checkpoint, builds an orthogonal projector onto the
row space of the encoder's LINEAR map W_effective = W_enc_2 @ W_enc_1, and runs
SEP-token activation patching restricted to that subspace:

    h_patched = P_mlp @ h_donor + (I - P_mlp) @ h_recipient

All donor/recipient generation, decoding, and outcome classification are reused
from probe_subspace_patching.py / activation_patching.py so results are directly
comparable to the 2D-probe, per-disk, and full-residual experiments.

LIMITATION: P_mlp is built from the encoder's linear component only; it ignores
the ReLU nonlinearity. The true set of directions the encoder can read is not a
linear subspace, so P_mlp is a linear approximation of the encoder's readable
subspace. This is stated in the run output.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.append(str(_REPO))

from model import MLPEncoder, effective_linear_map  # noqa: E402
from toh_transformer import activation_patching as ap  # noqa: E402
from toh_transformer import probe_subspace_patching as ps  # noqa: E402
from toh_transformer import utils as utils  # noqa: E402
from toh_transformer.data import Vocabulary  # noqa: E402

State = ap.State


def build_mlp_projector(encoder: MLPEncoder, device: torch.device):
    """P_mlp onto the row space of W_effective = W_enc_2 @ W_enc_1. Returns (P, rank)."""
    W_effective = effective_linear_map(encoder).to(device=device, dtype=torch.float32)  # (h_dim, 128)
    _, S, Vt = torch.linalg.svd(W_effective, full_matrices=False)
    threshold = float(S.max()) * 1e-6
    keep = S > threshold
    basis = Vt[keep]  # (rank, 128)
    rank = int(basis.shape[0])
    projector = basis.T @ basis  # (128, 128)
    return projector, rank


def check_projector(P: torch.Tensor) -> None:
    if not torch.allclose(P, P.T, atol=1e-5):
        raise AssertionError("P_mlp is not symmetric")
    if not torch.allclose(P @ P, P, atol=1e-4):
        raise AssertionError("P_mlp is not idempotent (P@P != P)")


def main() -> None:
    argp = argparse.ArgumentParser(description="MLP-encoder subspace activation patching")
    argp.add_argument("--checkpoint", required=True, help="trained joint-MLP .pt")
    argp.add_argument("--model_checkpoint", default="toh_transformer/checkpoints/n4/best.pt")
    argp.add_argument("--n_disks", type=int, default=4)
    argp.add_argument("--layer", type=int, default=5)
    argp.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    argp.add_argument("--seed", type=int, default=42)
    argp.add_argument("--max_pairs", type=int, default=0, help=">0 caps pairs (smoke test only)")
    argp.add_argument("--output", required=True)
    args = argp.parse_args()

    if args.n_disks != 4:
        raise ValueError("Defined for n_disks=4")
    utils.set_seed(args.seed)
    device = torch.device(args.device)
    vocab = Vocabulary()
    utils.confirm_tokenizer_mapping(vocab)

    # Rebuild the trained encoder from the checkpoint.
    ck = torch.load(args.checkpoint, map_location=device, weights_only=False)
    h_dim = int(ck["h_dim"])
    regime = ck["regime"]
    encoder = MLPEncoder(in_dim=128, hidden=64, h_dim=h_dim).to(device)
    encoder.load_state_dict(ck["encoder_state"])
    encoder.eval()

    projector, rank = build_mlp_projector(encoder, device)
    check_projector(projector)
    print("[NOTE] P_mlp uses only the encoder's LINEAR component (W_enc_2 @ W_enc_1); "
          "the ReLU nonlinearity is ignored, so P_mlp is a linear approximation of the "
          "encoder's readable subspace.")
    print(f"[INFO] regime={regime} h_dim={h_dim} subspace rank={rank} "
          f"({'== h_dim (full rank)' if rank == h_dim else '< h_dim (rank-deficient)'})")

    model = ap.load_model(utils.resolve_checkpoint_path(args.model_checkpoint, args.n_disks),
                          args.n_disks, device)
    goal: State = tuple(2 for _ in range(args.n_disks))
    all_states = ap.enumerate_states(args.n_disks)
    state_to_all_index = {s: i for i, s in enumerate(all_states)}
    sep2_idx = 2 * args.n_disks + 2
    if sep2_idx != 10:
        raise AssertionError(f"Expected SEP index 10, got {sep2_idx}")

    cached_sep_by_layer = ap.extract_sep2_activations(
        model=model, starts=all_states, goal=goal, vocab=vocab, device=device, batch_size=128)

    clean_by_state: Dict[State, Dict[str, object]] = {}
    valid_states: List[State] = []
    for st in all_states:
        gen_ids, eos = utils.greedy_decode_ids(
            model=model, context_ids=utils.build_context_ids(st, goal, vocab),
            eos_id=vocab.eos_id, device=device)
        parsed = ap.parse_generation(start=st, goal=goal, generated_ids=gen_ids, eos_seen=eos, vocab=vocab)
        clean_by_state[st] = parsed
        if bool(parsed["reaches_goal"]):
            valid_states.append(st)
    pairs = [(a, b) for a in valid_states for b in valid_states if a != b]
    if args.max_pairs and args.max_pairs > 0:
        pairs = pairs[: args.max_pairs]
        print(f"[WARN] SMOKE TEST: capping to {len(pairs)} pairs")
    print(f"[INFO] valid_states={len(valid_states)} pairs={len(pairs)}")

    # Sanity: injecting the recipient's own activation (h_d == h_r) is a no-op.
    unchanged, checked = ps.sanity_check_noop_patch(
        model=model, vocab=vocab, goal=goal, sep2_idx=sep2_idx, layer=args.layer,
        projector=projector, valid_states=valid_states, clean_by_state=clean_by_state,
        cached_sep_by_layer=cached_sep_by_layer, state_to_all_index=state_to_all_index, device=device)
    print(f"[SANITY] no-op patch unchanged {unchanged}/{checked}")
    if unchanged != checked:
        raise AssertionError("Sanity check failed: no-op MLP patch changed the decode")

    rows = ps.run_layer_probe_patch(
        model=model, vocab=vocab, goal=goal, sep2_idx=sep2_idx, layer=args.layer,
        projector=projector, pairs=pairs, clean_by_state=clean_by_state,
        cached_sep_by_layer=cached_sep_by_layer, state_to_all_index=state_to_all_index, device=device)
    summ = ps.summarize_to_schema(rows)

    result = {
        "config": {
            "probe_checkpoint": str(args.checkpoint),
            "model_checkpoint": str(args.model_checkpoint),
            "regime": regime,
            "layer": args.layer,
            "h_dim": h_dim,
            "n_donor_recipient_pairs": len(pairs),
            "uses_linear_component_only": True,
        },
        "regime": regime,
        "layer": args.layer,
        "h_dim": h_dim,
        "subspace_dim": rank,
        "n_pairs": summ["n_pairs"],
        "full_transfer_pct": summ["full_transfer_pct"],
        "partial_pct": summ["partial_pct"],
        "partial_mean_k": summ["partial_mean_k"],
        "unchanged_pct": summ["unchanged_pct"],
        "novel_correct_pct": summ["novel_correct_pct"],
        "disrupted_pct": summ["disrupted_pct"],
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"[INFO] rank={rank} full={summ['full_transfer_pct']:.2f}% partial={summ['partial_pct']:.2f}% "
          f"unchanged={summ['unchanged_pct']:.2f}% disrupted={summ['disrupted_pct']:.2f}%")
    print(f"[INFO] Wrote {out}")


if __name__ == "__main__":
    main()
