#!/usr/bin/env python3
"""Load all joint-MLP probe + patching JSONs and print the two comparison tables.

Also loads the prior surgical-patching results (2D distance probe, per-disk union)
and the full-residual baseline so the patching table is directly comparable.

Dependencies: torch not required (pure json/stdlib).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

PROBE_DIR = Path("outputs/joint_mlp_probes")
PS_2D_JSON = Path("toh_transformer/patching_output/probe_subspace_patching.json")
PERDISK_JSON = Path("toh_transformer/patching_output/perdisk_subspace_patching.json")
FULL_RESIDUAL_JSON = Path("toh_transformer/activation_patching_output/activation_patching_results.json")

# Order used in both tables.
REGIME_ORDER = ["sequential_cls_first", "joint", "sequential_dist_first"]
REGIME_LABEL = {
    "sequential_cls_first": "sequential cls-first",
    "joint": "joint",
    "sequential_dist_first": "sequential dist-first",
}
PROBE_ROW_LABEL = {
    "sequential_cls_first": "Sequential (cls-first)",
    "joint": "Joint",
    "sequential_dist_first": "Sequential (dist-first)",
}

CAT = {
    "full": "FULL_TRANSFER", "partial": "PARTIAL_TRANSFER", "unch": "RECIPIENT_UNCHANGED",
    "novel": "NOVEL_CORRECT", "disr": "NOVEL_INCORRECT",
}


def load_json(p: Path) -> Optional[dict]:
    if not p.exists():
        print(f"[WARN] missing: {p}")
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def full_residual_row(layer: int) -> Optional[Dict[str, float]]:
    data = load_json(FULL_RESIDUAL_JSON)
    if data is None:
        return None
    rows = [r for r in data if int(r["layer"]) == layer]
    if not rows:
        return None
    n = len(rows)
    counts = {v: 0 for v in CAT.values()}
    ks: List[int] = []
    for r in rows:
        counts[str(r["category"])] += 1
        if r["category"] == CAT["partial"] and r.get("partial_k") is not None:
            ks.append(int(r["partial_k"]))
    return {
        "full_transfer_pct": 100.0 * counts[CAT["full"]] / n,
        "partial_pct": 100.0 * counts[CAT["partial"]] / n,
        "partial_mean_k": (sum(ks) / len(ks)) if ks else 0.0,
        "unchanged_pct": 100.0 * counts[CAT["unch"]] / n,
        "novel_correct_pct": 100.0 * counts[CAT["novel"]] / n,
        "disrupted_pct": 100.0 * counts[CAT["disr"]] / n,
    }


def prior_row(js: Optional[dict], layer: int) -> Optional[dict]:
    if js is None:
        return None
    rl = js.get("results_by_layer", {})
    return rl.get(str(layer))


def print_probe_table(layer: int, h_dims) -> None:
    hdr = ("Regime                    | h_dim | Spearman | Pearson | NS acc | "
           "Disk 0 | Disk 1 | Disk 2 | Disk 3 | Mean")
    sep = ("--------------------------+-------+----------+---------+--------+"
           "--------+--------+--------+--------+------")
    print(hdr)
    print(sep)
    for regime in REGIME_ORDER:
        for h in h_dims:
            m = load_json(PROBE_DIR / f"regime_{regime}_layer{layer}_h{h}.json")
            label = PROBE_ROW_LABEL[regime]
            if m is None:
                print(f"{label:<25s} | {h:>4d}  | {'--':>8s} | {'--':>7s} | "
                      f"{'--':>6s} | {'--':>6s} | {'--':>6s} | {'--':>6s} | {'--':>6s} | {'--':>6s}")
                continue
            d = m["per_disk_acc"]
            print(f"{label:<25s} | {h:>4d}  | {m['spearman']:>8.3f} | {m['pearson']:>7.3f} | "
                  f"{100*m['nearest_state_acc']:>5.2f}% | {100*d[0]:>5.2f}% | {100*d[1]:>5.2f}% | "
                  f"{100*d[2]:>5.2f}% | {100*d[3]:>5.2f}% | {100*m['mean_disk_acc']:>5.2f}%")


def _patch_row(label: str, dim, m: dict) -> str:
    dim_s = f"{dim:>3}" if isinstance(dim, int) else f"{dim:>3}"
    return (f"{5:<5d} | {label:<37s} | {dim_s} | {m['full_transfer_pct']:>5.2f}% | "
            f"{m['partial_pct']:>5.2f}% ({m['partial_mean_k']:.2f}) | {m['unchanged_pct']:>6.2f}%    | "
            f"{m['novel_correct_pct']:>5.2f}% | {m['disrupted_pct']:>5.2f}%")


def print_patching_table(layer: int, h_dims) -> None:
    hdr = ("Layer | Method                                | Dim | Full  | Partial (K)   | "
           "Unchanged | Novel | Disrupted")
    sep = ("------+---------------------------------------+-----+-------+---------------+"
           "-----------+-------+----------")
    print(hdr)
    print(sep)

    # 2D distance probe
    r = prior_row(load_json(PS_2D_JSON), layer)
    if r:
        print(_patch_row("Probe subspace (2D)", 2, r))
    # per-disk union
    pr = load_json(PERDISK_JSON)
    r = prior_row(pr, layer)
    if r:
        dim = r.get("subspace_dim", 8)
        print(_patch_row("Per-disk subspace", dim, r))

    # MLP rows grouped by h_dim then regime
    for h in h_dims:
        for regime in REGIME_ORDER:
            m = load_json(PROBE_DIR / f"patching_{regime}_layer{layer}_h{h}.json")
            if m is None:
                continue
            label = f"MLP h={h} ({REGIME_LABEL[regime]})"
            print(_patch_row(label, m.get("subspace_dim", h), m))

    # full residual
    fr = full_residual_row(layer)
    if fr:
        print(_patch_row("Full residual", 128, fr))


INTERP = """
Interpretation of probe quality:
- Regime A (cls-first): tests whether a classification-optimised encoder incidentally
  supports 2D geometry. Expect Spearman < 0.94 if the two objectives are in tension.
- Regime B (joint): tests whether both objectives can be simultaneously optimised.
  Expect near-optimal performance on both if the objectives are compatible.
- Regime C (dist-first): tests whether a geometry-optimised encoder incidentally
  supports per-disk decodability. Expect per-disk accuracy to drop from 100% if
  the two objectives are in tension.

Interpretation of patching:
- Nine intervention granularities from 2 to 128 dimensions.
- If joint-regime patching outperforms both sequential regimes at the same h_dim,
  the joint bottleneck concentrates more causal information than either single-objective
  encoder.
- If dist-first patching outperforms cls-first patching at the same h_dim, the
  geometric encoding is more causally relevant than the classification encoding.
- Comparison to per-disk-union (8 dims) shows whether a jointly-trained bottleneck
  at 8 dims captures more causal signal than the union of independently-trained
  per-disk probes.
"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", type=int, default=5)
    ap.add_argument("--h_dims", default="8,16,32,64",
                    help="Comma/space-separated bottleneck sizes to include in the tables.")
    args = ap.parse_args()
    h_dims = [int(x) for x in args.h_dims.replace(",", " ").split()]

    print("=" * 100)
    print(f"PROBE QUALITY (layer {args.layer}, h_dims={h_dims})")
    print("=" * 100)
    print_probe_table(args.layer, h_dims)
    print()
    print("=" * 100)
    print(f"ACTIVATION PATCHING COMPARISON (layer {args.layer}, h_dims={h_dims})")
    print("=" * 100)
    print_patching_table(args.layer, h_dims)
    print(INTERP)


if __name__ == "__main__":
    main()
