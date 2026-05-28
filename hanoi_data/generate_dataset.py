"""Generate templated reasoning traces for every (s_I, s_G) pair with
s_I != s_G on 4-disk Tower of Hanoi.

Output: hanoi_data/train.jsonl and hanoi_data/test.jsonl (80/20 stratified
by optimal solution length, fixed seed for reproducibility).
"""

from __future__ import annotations

import json
import random
import sys
from collections import Counter, defaultdict
from itertools import product
from pathlib import Path
from typing import Dict, List

_HERE = Path(__file__).resolve().parent
if str(_HERE.parent) not in sys.path:
    sys.path.insert(0, str(_HERE.parent))

from hanoi_data.template import (
    N_DISKS,
    bfs_optimal,
    enumerate_states,
    generate_trace,
)

SEED = 42
TRAIN_FRAC = 0.8


def build_all_examples() -> List[Dict]:
    states = enumerate_states(N_DISKS)
    examples: List[Dict] = []
    for s_I, s_G in product(states, states):
        if s_I == s_G:
            continue
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


def main() -> None:
    print(f"[INFO] Building all examples (n_disks={N_DISKS}, "
          f"expecting 3^{N_DISKS} * (3^{N_DISKS} - 1) = "
          f"{3**N_DISKS * (3**N_DISKS - 1)} pairs)…")
    examples = build_all_examples()
    print(f"[INFO] Generated {len(examples)} examples")

    train, test = stratified_split(examples, TRAIN_FRAC, SEED)
    out_dir = _HERE
    train_path = out_dir / "train.jsonl"
    test_path = out_dir / "test.jsonl"
    write_jsonl(train_path, train)
    write_jsonl(test_path, test)
    print(f"[INFO] Wrote {train_path}  ({len(train)} rows)")
    print(f"[INFO] Wrote {test_path}   ({len(test)} rows)")

    print_stats("TRAIN", train)
    print_stats("TEST", test)


if __name__ == "__main__":
    main()
