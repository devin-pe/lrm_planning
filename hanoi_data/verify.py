"""Sanity-check the generated train.jsonl / test.jsonl.

Run after generate_dataset.py.
"""

from __future__ import annotations

import ast
import json
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List

_HERE = Path(__file__).resolve().parent
if str(_HERE.parent) not in sys.path:
    sys.path.insert(0, str(_HERE.parent))

from hanoi_data.template import apply_move

# Substrings that must never overlap a supervision span.
FORBIDDEN_MARKERS = ("<think>", "</think>", "Move ", "State:", "->", "moves =")

TOTAL_EXPECTED = 6480  # 3^4 * (3^4 - 1)
TRAIN_FRAC = 0.8
SPLIT_TOLERANCE = 1     # per-stratum |train_size - round(L*0.8)| <= 1


def load_jsonl(path: Path) -> List[Dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def simulate(s_I: List[int], moves: List[List[int]]) -> tuple:
    state = tuple(int(x) for x in s_I)
    for m in moves:
        # Moves are 3-tuples [disk, from, to]. Disk is informational; the
        # apply_move routine derives it from the topmost source-peg disk.
        _, fp, tp = m
        state = apply_move(state, int(fp), int(tp))
    return state


def verify_example(row: Dict) -> None:
    completion = row["completion"]
    s_I = row["s_I"]
    s_G = row["s_G"]
    moves = row["moves"]
    sups = row["supervisions"]

    # 1. Each supervision substring parses to the target state.
    for cs, ce, target in sups:
        substr = completion[cs:ce]
        parsed = ast.literal_eval(substr)
        assert isinstance(parsed, list) and tuple(parsed) == tuple(target), (
            f"Supervision [{cs}:{ce}]={substr!r} does not match target {target}"
        )
        # 3. No forbidden markers inside the span.
        for marker in FORBIDDEN_MARKERS:
            assert marker not in substr, (
                f"Supervision span {substr!r} contains forbidden marker {marker!r}"
            )

    # Simulate moves; final state must equal s_G.
    final = simulate(s_I, moves)
    assert final == tuple(s_G), (
        f"Replaying moves from {s_I} ended at {final}, expected {s_G}"
    )

    # Each entry in moves is a 2-tuple.
    for m in moves:
        assert len(m) == 3, f"Move {m} is not [disk, from, to]"


def check_no_overlap_with_other_spans(row: Dict) -> None:
    """Ensure the supervised character ranges are disjoint and ordered."""
    last_end = -1
    for cs, ce, _ in row["supervisions"]:
        assert cs >= last_end, f"Overlapping supervision spans: {row['supervisions']}"
        last_end = ce


def print_example(label: str, row: Dict) -> None:
    print("=" * 80)
    print(f"{label}   s_I={row['s_I']}  s_G={row['s_G']}  L={len(row['moves'])}")
    print("=" * 80)
    print("SYSTEM PROMPT")
    print(row["system_prompt"])
    print("USER PROMPT")
    print(row["user_prompt"])
    print("COMPLETION")
    print(row["completion"])
    print("SUPERVISIONS")
    for cs, ce, t in row["supervisions"]:
        print(f"  [{cs:4d}:{ce:4d}] = {row['completion'][cs:ce]!r}  target={t}")
    print()


def check_stratification(train: List[Dict], test: List[Dict]) -> None:
    combined = Counter(len(r["moves"]) for r in train + test)
    train_counts = Counter(len(r["moves"]) for r in train)
    for L, total in combined.items():
        expected_train = round(total * TRAIN_FRAC)
        diff = abs(train_counts[L] - expected_train)
        assert diff <= SPLIT_TOLERANCE, (
            f"Stratum L={L}: total={total}, train={train_counts[L]}, "
            f"expected≈{expected_train} (tol={SPLIT_TOLERANCE})"
        )


def main() -> None:
    train_path = _HERE / "train.jsonl"
    test_path = _HERE / "test.jsonl"
    train = load_jsonl(train_path)
    test = load_jsonl(test_path)

    # 4. total count
    total = len(train) + len(test)
    assert total == TOTAL_EXPECTED, (
        f"Expected {TOTAL_EXPECTED} examples, got {total} (train={len(train)}, "
        f"test={len(test)})"
    )
    print(f"[OK] total examples = {total}")

    # 5. stratified split balance
    check_stratification(train, test)
    print(f"[OK] stratified 80/20 split per solution length "
          f"(tol={SPLIT_TOLERANCE} examples per stratum)")

    # 2. Sample-level checks + printing.
    rng = random.Random(0)
    train_sample = rng.sample(train, 10)
    test_sample = rng.sample(test, 10)
    for r in train_sample + test_sample:
        verify_example(r)
        check_no_overlap_with_other_spans(r)
    print(f"[OK] verified spans + replay on 20 random examples")

    # Run the same checks across the FULL set (cheap; ~6.5k rows).
    for r in train + test:
        verify_example(r)
        check_no_overlap_with_other_spans(r)
    print(f"[OK] verified spans + replay across all {total} examples")

    for i, r in enumerate(train_sample[:3], start=1):
        print_example(f"TRAIN sample #{i}", r)
    for i, r in enumerate(test_sample[:3], start=1):
        print_example(f"TEST sample #{i}", r)


if __name__ == "__main__":
    main()
