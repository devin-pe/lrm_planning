"""Side-by-side comparison of two run summaries.

The Position-B/C drift delta is the H3-relevant indicator (a successful
probe-loss regime should encode state more faithfully at the move-emission
positions than the baseline).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple


def _load(path: str) -> Dict:
    return json.loads(Path(path).read_text())


def _pct(num: float, denom: float) -> str:
    if denom == 0:
        return "0.0%"
    return f"{100.0 * num / denom:.1f}%"


def _solve_correct(summary: Dict) -> Tuple[int, int]:
    s = summary.get("solve")
    if not s:
        return 0, 0
    cc = s["category_counts"]
    return cc.get("Optimal", 0) + cc.get("Suboptimal", 0), s["n"]


def _fmt_int_delta(a: int, b: int) -> str:
    d = b - a
    return f"{d:+d}"


def _fmt_float_delta(a: Optional[float], b: Optional[float]) -> str:
    if a is None or b is None:
        return "n/a"
    return f"{b - a:+.4f}"


def main() -> None:
    p = argparse.ArgumentParser(description="Compare two eval summaries")
    p.add_argument("--summary_json", action="append", required=True,
                   help="Pass twice: --summary_json A.json --summary_json B.json")
    args = p.parse_args()
    if len(args.summary_json) != 2:
        raise SystemExit("Need exactly two --summary_json args (baseline first, probe second).")

    a = _load(args.summary_json[0])
    b = _load(args.summary_json[1])
    label_a = a.get("train_config", {}).get("regime", "A")
    label_b = b.get("train_config", {}).get("regime", "B")

    print("=" * 95)
    print(f"COMPARISON   A = {args.summary_json[0]}  (regime={label_a})")
    print(f"             B = {args.summary_json[1]}  (regime={label_b})")
    print("=" * 95)
    print(f"{'Metric':<42} {'A':>16} {'B':>16} {'Δ (B-A)':>17}")
    print("-" * 95)

    a_correct, a_n = _solve_correct(a)
    b_correct, b_n = _solve_correct(b)
    print(f"{'Solve rate (Optimal+Suboptimal)':<42} "
          f"{f'{a_correct}/{a_n} ({_pct(a_correct, a_n)})':>16} "
          f"{f'{b_correct}/{b_n} ({_pct(b_correct, b_n)})':>16} "
          f"{_fmt_int_delta(a_correct, b_correct):>17}")

    for cat in ("Optimal", "Suboptimal", "Incorrect", "Illegal"):
        av = a.get("solve", {}).get("category_counts", {}).get(cat, 0)
        bv = b.get("solve", {}).get("category_counts", {}).get(cat, 0)
        print(f"  {cat:<40} {av:>16d} {bv:>16d} {_fmt_int_delta(av, bv):>17}")

    print("-" * 95)
    print("Fresh per-layer probe metrics:")
    fresh_a = a.get("probe", {}).get("fresh", {})
    fresh_b = b.get("probe", {}).get("fresh", {})
    keys = sorted(set(fresh_a.keys()) | set(fresh_b.keys()))
    for k in keys:
        ma = fresh_a.get(k, {})
        mb = fresh_b.get(k, {})
        for metric in ("spearman", "pearson", "adj_acc"):
            va = ma.get(metric); vb = mb.get(metric)
            print(f"  {k} {metric:<8} {str(va)[:8]:>16} {str(vb)[:8]:>16} "
                  f"{_fmt_float_delta(va, vb):>17}")

    print("-" * 95)
    print("Training-time probe (at probe-regime's trained layer):")
    th_a = a.get("probe", {}).get("trained_head", {})
    th_b = b.get("probe", {}).get("trained_head", {})
    for k in sorted(set(th_a.keys()) | set(th_b.keys())):
        ma = th_a.get(k, {})
        mb = th_b.get(k, {})
        for metric in ("spearman", "pearson"):
            va = ma.get(metric); vb = mb.get(metric)
            print(f"  {k} {metric:<8} {str(va)[:8]:>16} {str(vb)[:8]:>16} "
                  f"{_fmt_float_delta(va, vb):>17}")

    # Headline assessment
    print("=" * 95)
    print("HEADLINE")
    if a_n > 0 and b_n > 0:
        if b_correct > a_correct:
            print(f"  [+] probe regime improves solve rate "
                  f"({_pct(a_correct, a_n)} → {_pct(b_correct, b_n)})")
        else:
            print(f"  [-] probe regime does NOT improve solve rate "
                  f"({_pct(a_correct, a_n)} → {_pct(b_correct, b_n)})")

    pos_b_a = fresh_a.get(f"L{a.get('probe', {}).get('train_layer', 36)}_B", {}).get("spearman")
    pos_b_b = fresh_b.get(f"L{b.get('probe', {}).get('train_layer', 36)}_B", {}).get("spearman")
    if pos_b_a is not None and pos_b_b is not None:
        if pos_b_b > pos_b_a:
            print(f"  [+] Position B drift reduced (Spearman {pos_b_a:.3f} → {pos_b_b:.3f})")
        else:
            print(f"  [-] Position B drift NOT reduced (Spearman {pos_b_a:.3f} → {pos_b_b:.3f})")

    pos_c_a = fresh_a.get(f"L{a.get('probe', {}).get('train_layer', 36)}_C", {}).get("spearman")
    pos_c_b = fresh_b.get(f"L{b.get('probe', {}).get('train_layer', 36)}_C", {}).get("spearman")
    if pos_c_a is not None and pos_c_b is not None:
        if pos_c_b > pos_c_a:
            print(f"  [+] Position C drift reduced (Spearman {pos_c_a:.3f} → {pos_c_b:.3f})")
        else:
            print(f"  [-] Position C drift NOT reduced (Spearman {pos_c_a:.3f} → {pos_c_b:.3f})")

    print("=" * 95)


if __name__ == "__main__":
    main()
