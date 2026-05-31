"""Generate templated reasoning traces for Tower of Hanoi problems.

For n_disks == 4 (default): enumerate ALL 3^4 × (3^4 - 1) = 6480 (s_I, s_G)
pairs, do an 80/20 stratified split, and write `train.jsonl` + `test.jsonl`.

For n_disks ∈ {3, 5} (OOD evaluation sets): produce a single TEST file
`test_n{N}.jsonl` — no training data needed for these disk counts.
  - n=3: enumerate all 3^3 × (3^3 - 1) = 702 pairs (small enough).
  - n=5: random-sample `--max_problems` pairs (default 1296, matching the
    4-disk test count) since the full 58806 pairs would be wasteful.

Output naming is preserved exactly for n=4 (back-compat with downstream).
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from itertools import product
from pathlib import Path
from typing import Dict, List, Optional, Tuple

_HERE = Path(__file__).resolve().parent
if str(_HERE.parent) not in sys.path:
    sys.path.insert(0, str(_HERE.parent))

from hanoi_data.template import (
    bfs_optimal,
    enumerate_states,
    generate_trace,
)

SEED = 42
TRAIN_FRAC = 0.8


def build_all_examples(n_disks: int) -> List[Dict]:
    states = enumerate_states(n_disks=n_disks)
    examples: List[Dict] = []
    for s_I, s_G in product(states, states):
        if s_I == s_G:
            continue
        moves = bfs_optimal(s_I, s_G)
        examples.append(generate_trace(s_I, s_G, moves))
    return examples


def build_sampled_examples(n_disks: int, max_problems: int, seed: int) -> List[Dict]:
    """Random sample (s_I, s_G) pairs, then BFS each. Used when the full
    Cartesian product is too large to enumerate (n_disks=5 → 58806 pairs).
    """
    states = enumerate_states(n_disks=n_disks)
    rng = random.Random(seed)
    pairs: List[Tuple] = []
    seen = set()
    while len(pairs) < max_problems:
        i = rng.randrange(len(states))
        j = rng.randrange(len(states))
        if i == j:
            continue
        key = (i, j)
        if key in seen:
            continue
        seen.add(key)
        pairs.append((states[i], states[j]))
    examples: List[Dict] = []
    for s_I, s_G in pairs:
        moves = bfs_optimal(s_I, s_G)
        examples.append(generate_trace(s_I, s_G, moves))
    return examples


def stratified_split(
    examples: List[Dict],
    train_frac: float,
    seed: int,
) -> tuple[List[Dict], List[Dict]]:
    rng = random.Random(seed)
    by_len: Dict[int, List[Dict]] = defaultdict(list)
    for ex in examples:
        by_len[len(ex["moves"])].append(ex)

    train: List[Dict] = []
    test: List[Dict] = []
    for L in sorted(by_len.keys()):
        bucket = by_len[L][:]
        rng.shuffle(bucket)
        n_train = round(len(bucket) * train_frac)
        train.extend(bucket[:n_train])
        test.extend(bucket[n_train:])
    rng.shuffle(train)
    rng.shuffle(test)
    return train, test


def write_jsonl(path: Path, rows: List[Dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def print_stats(label: str, rows: List[Dict]) -> None:
    n = len(rows)
    lens = Counter(len(r["moves"]) for r in rows)
    comp_chars = [len(r["completion"]) for r in rows]
    sup_counts = [len(r["supervisions"]) for r in rows]
    mean_chars = sum(comp_chars) / max(n, 1)
    max_chars = max(comp_chars) if comp_chars else 0
    mean_sup = sum(sup_counts) / max(n, 1)
    print(f"\n[{label}]  n = {n}")
    print(f"  completion chars   mean={mean_chars:8.1f}  max={max_chars}  "
          f"~tokens(/4)  mean={mean_chars / 4:7.1f}  max={max_chars / 4:.0f}")
    print(f"  supervised spans   mean per example = {mean_sup:.2f}")
    print("  optimal-length distribution:")
    for L in sorted(lens):
        print(f"    L={L:2d}  count={lens[L]:4d}")


def _test_path(n_disks: int, out_dir: Path) -> Path:
    """4-disk uses the legacy `test.jsonl` name; OOD splits use `test_n{N}.jsonl`."""
    return out_dir / ("test.jsonl" if n_disks == 4 else f"test_n{n_disks}.jsonl")


def main() -> None:
    p = argparse.ArgumentParser(description="Generate ToH dataset (train/test JSONL)")
    p.add_argument("--n_disks", type=int, choices=[3, 4, 5], default=4)
    p.add_argument("--max_problems", type=int, default=None,
                   help="Cap on random-sampled pairs (for n_disks=5). Ignored "
                        "for n_disks=3 (all 702 pairs enumerated).")
    p.add_argument("--seed", type=int, default=SEED)
    args = p.parse_args()

    n = args.n_disks
    full_pairs = 3 ** n * (3 ** n - 1)

    if n == 4:
        # Legacy 4-disk flow: enumerate-all + 80/20 stratified split.
        print(f"[INFO] Building all examples (n_disks={n}, expecting "
              f"3^{n} * (3^{n} - 1) = {full_pairs} pairs)…")
        examples = build_all_examples(n_disks=n)
        print(f"[INFO] Generated {len(examples)} examples")

        train, test = stratified_split(examples, TRAIN_FRAC, args.seed)
        train_path = _HERE / "train.jsonl"
        test_path = _test_path(n, _HERE)
        write_jsonl(train_path, train)
        write_jsonl(test_path, test)
        print(f"[INFO] Wrote {train_path}  ({len(train)} rows)")
        print(f"[INFO] Wrote {test_path}   ({len(test)} rows)")
        print_stats("TRAIN", train)
        print_stats("TEST", test)
        return

    # n_disks ∈ {3, 5}: OOD test-only.
    if n == 3:
        print(f"[INFO] n=3: enumerating all {full_pairs} pairs")
        examples = build_all_examples(n_disks=n)
    else:  # n == 5
        cap = args.max_problems if args.max_problems is not None else 1296
        print(f"[INFO] n=5: sampling {cap} of {full_pairs} possible pairs (seed={args.seed})")
        examples = build_sampled_examples(n_disks=n, max_problems=cap, seed=args.seed)

    test_path = _test_path(n, _HERE)
    write_jsonl(test_path, examples)
    print(f"[INFO] Wrote {test_path}  ({len(examples)} rows)")
    print_stats(f"TEST (n_disks={n})", examples)


if __name__ == "__main__":
    main()
