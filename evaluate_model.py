#!/usr/bin/env python3
"""Classify and analyze LLM failures on 81 flat-to-flat ToH problems.

Loads generated_texts.json, re-parses moves permissively, simulates each
sequence, classifies errors, and produces CoT analysis tables.

Works with any model whose outputs were collected into generated_texts.json
(Qwen3-27B, DeepSeek-R1-Distill-Qwen-32B, etc.).  Point --generated_texts
and --output_dir at the right directories and it just works.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from itertools import product
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

sys.path.insert(0, str(Path(__file__).parent))
from planning import TowersOfHanoiSolver

State = Tuple[int, ...]

# ── Error categories ──────────────────────────────────────────────────────────
CORRECT_OPTIMAL         = "CORRECT_OPTIMAL"
CORRECT_SUBOPTIMAL      = "CORRECT_SUBOPTIMAL"
PARSE_ERROR             = "PARSE_ERROR"
ILLEGAL_EMPTY_SOURCE    = "ILLEGAL_EMPTY_SOURCE"
ILLEGAL_LARGER_ON_SMALLER = "ILLEGAL_LARGER_ON_SMALLER"
ILLEGAL_WRONG_DISK      = "ILLEGAL_WRONG_DISK"
ILLEGAL_BAD_FORMAT      = "ILLEGAL_BAD_FORMAT"
WRONG_GOAL_STATE        = "WRONG_GOAL_STATE"
PREMATURE_STOP          = "PREMATURE_STOP"
EXCESSIVE_MOVES         = "EXCESSIVE_MOVES"

ALL_CATEGORIES = [
    CORRECT_OPTIMAL, CORRECT_SUBOPTIMAL, PARSE_ERROR,
    ILLEGAL_EMPTY_SOURCE, ILLEGAL_LARGER_ON_SMALLER, ILLEGAL_WRONG_DISK, ILLEGAL_BAD_FORMAT,
    WRONG_GOAL_STATE, PREMATURE_STOP, EXCESSIVE_MOVES,
]
ILLEGAL_CATEGORIES = {ILLEGAL_EMPTY_SOURCE, ILLEGAL_LARGER_ON_SMALLER,
                      ILLEGAL_WRONG_DISK, ILLEGAL_BAD_FORMAT}
CORRECT_CATEGORIES = {CORRECT_OPTIMAL, CORRECT_SUBOPTIMAL}

STRATEGY_KEYWORDS = [
    "recursive", "recursively", "sub-problem", "sub problem", "subproblem",
    "auxiliary peg", "peg as auxiliary", "clear the way", "largest disk first",
    "move disk 4 first", "bottom disk", "standard hanoi", "standard tower",
    "2^n", "2**n", "n-1 disk", "divide and conquer",
    "base case", "recursive call", "helper function",
]


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Analyse LLM ToH failures")
    p.add_argument("--generated_texts", required=True,
                   help="Path to generated_texts.json produced by generate.py")
    p.add_argument("--n_disks", type=int, default=4)
    p.add_argument("--output_dir", required=True,
                   help="Directory to write evaluate_results.json into")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


# ── State helpers ─────────────────────────────────────────────────────────────

def enumerate_states(n_disks: int) -> List[State]:
    return [tuple(s) for s in product(range(3), repeat=n_disks)]


def state_tuple_to_pegs(state: State, n_disks: int) -> List[List[int]]:
    """Convert compact state tuple to list-of-pegs (bottom→top per peg)."""
    pegs: List[List[int]] = [[], [], []]
    for disk in range(n_disks, 0, -1):   # largest first → appended below smaller ones
        peg = int(state[disk - 1])
        pegs[peg].append(disk)
    return pegs


def pegs_to_state(pegs: List[List[int]], n_disks: int) -> State:
    state = [0] * n_disks
    for peg_idx, peg in enumerate(pegs):
        for disk in peg:
            state[disk - 1] = peg_idx
    return tuple(state)


def pegs_copy(pegs: List[List[int]]) -> List[List[int]]:
    return [p[:] for p in pegs]


def pegs_equal(a: List[List[int]], b: List[List[int]]) -> bool:
    return a == b


# ── BFS optimal length ────────────────────────────────────────────────────────

def compute_opt_lengths(states: List[State], n_disks: int) -> Dict[State, int]:
    solver = TowersOfHanoiSolver()
    goal_pegs = [[], [], list(range(n_disks, 0, -1))]
    goal_state = tuple([2] * n_disks)
    lengths: Dict[State, int] = {}
    for s in states:
        if s == goal_state:
            lengths[s] = 0
            continue
        init_pegs = state_tuple_to_pegs(s, n_disks)
        moves = solver.solve(num_disks=n_disks, initial_state=init_pegs, goal_state=goal_pegs)
        lengths[s] = len(moves) if moves is not None else -1
    return lengths


# ── Move parsing ──────────────────────────────────────────────────────────────

def _find_last_complete_moves_block(text: str) -> Optional[str]:
    """Return the last complete `moves = [...]` block in text."""
    last_ok: Optional[str] = None
    for m in re.finditer(r'moves\s*=\s*\[', text, re.IGNORECASE):
        bracket_start = text.find('[', m.start())
        if bracket_start < 0:
            continue
        depth = 0
        for idx in range(bracket_start, len(text)):
            c = text[idx]
            if c == '[':
                depth += 1
            elif c == ']':
                depth -= 1
                if depth == 0:
                    candidate = text[bracket_start:idx + 1].strip()
                    if candidate != '[]':
                        last_ok = candidate
                    break
    return last_ok


def _parse_moves_json(block: str) -> Optional[List[List[int]]]:
    cleaned = re.sub(r'#[^\n]*', '', block)           # strip inline comments
    cleaned = re.sub(r',\s*(\])', r'\1', cleaned)     # trailing commas
    try:
        parsed = json.loads(cleaned)
        if not isinstance(parsed, list):
            return None
        if not parsed:
            return None
        # Validate each entry is a list of 3 things
        for item in parsed:
            if not isinstance(item, list) or len(item) != 3:
                return None
        return [[int(x) for x in item] for item in parsed]
    except (json.JSONDecodeError, ValueError, TypeError):
        return None


def parse_moves(text: str) -> Optional[List[List[int]]]:
    block = _find_last_complete_moves_block(text)
    if block is None:
        return None
    return _parse_moves_json(block)


# ── Move simulation ───────────────────────────────────────────────────────────

def simulate_single_move(
    pegs: List[List[int]],
    move: Any,
    n_disks: int,
) -> Tuple[str, Optional[str]]:
    """Apply one move in-place. Returns ('OK', None) or ('ERROR', category)."""
    if not isinstance(move, list) or len(move) != 3:
        return "ERROR", ILLEGAL_BAD_FORMAT
    try:
        disk, from_peg, to_peg = int(move[0]), int(move[1]), int(move[2])
    except (TypeError, ValueError):
        return "ERROR", ILLEGAL_BAD_FORMAT

    if disk < 1 or disk > n_disks or from_peg < 0 or from_peg > 2 or to_peg < 0 or to_peg > 2:
        return "ERROR", ILLEGAL_BAD_FORMAT
    if not pegs[from_peg]:
        return "ERROR", ILLEGAL_EMPTY_SOURCE
    if pegs[from_peg][-1] != disk:
        return "ERROR", ILLEGAL_WRONG_DISK
    if pegs[to_peg] and pegs[to_peg][-1] < disk:
        return "ERROR", ILLEGAL_LARGER_ON_SMALLER

    pegs[to_peg].append(pegs[from_peg].pop())
    return "OK", None


def simulate_moves(
    start_pegs: List[List[int]],
    moves: List[List[int]],
    goal_pegs: List[List[int]],
    n_disks: int,
    opt_len: int,
) -> Dict[str, Any]:
    """Simulate full move sequence and return classification + error metadata."""
    pegs = pegs_copy(start_pegs)
    legal_prefix = 0
    error_step: Optional[int] = None
    error_category: Optional[str] = None
    state_at_error: Optional[Tuple] = None

    for step_idx, move in enumerate(moves):
        status, cat = simulate_single_move(pegs, move, n_disks)
        if status == "ERROR":
            error_step = step_idx + 1
            error_category = cat
            state_at_error = pegs_to_state(pegs, n_disks)
            break
        legal_prefix += 1

    reached_goal = pegs_equal(pegs, goal_pegs)
    n_moves = len(moves)

    if error_step is not None:
        category = error_category
    elif reached_goal:
        if n_moves == opt_len:
            category = CORRECT_OPTIMAL
        else:
            category = CORRECT_SUBOPTIMAL
    elif n_moves < opt_len:
        category = PREMATURE_STOP
    elif n_moves > opt_len:
        category = EXCESSIVE_MOVES
    else:
        category = WRONG_GOAL_STATE

    error_fraction = (error_step / n_moves) if (error_step is not None and n_moves > 0) else None

    return {
        "category": category,
        "error_step": error_step,
        "error_fraction": error_fraction,
        "legal_prefix_length": legal_prefix,
        "state_at_error": list(state_at_error) if state_at_error else None,
        "n_moves": n_moves,
    }


# ── CoT analysis ──────────────────────────────────────────────────────────────

_INTERMEDIATE_STATE_RE = re.compile(
    r'[Ss]tate\s*:?\s*[Pp]0\s*[=:]\s*`?\[([^\]]*)\]`?'
    r'\s*,\s*[Pp]1\s*[=:]\s*`?\[([^\]]*)\]`?'
    r'\s*,\s*[Pp]2\s*[=:]\s*`?\[([^\]]*)\]`?',
    re.DOTALL,
)

_PEG_DESC_RE = re.compile(
    r'[Pp]eg\s+0\s*:\s*(.+?)(?:\n|[Pp]eg)',
    re.DOTALL,
)


def count_intermediate_states(text: str) -> int:
    """Count how many intermediate state descriptions appear in text."""
    return len(_INTERMEDIATE_STATE_RE.findall(text))


def has_strategic_language(text: str) -> bool:
    tl = text.lower()
    return any(kw in tl for kw in STRATEGY_KEYWORDS)


def cot_length(text: str) -> int:
    """CoT length = length of everything before the last 'moves = [...]' block."""
    block = _find_last_complete_moves_block(text)
    if block is None:
        return len(text)
    last_pos = text.rfind(block)
    return last_pos if last_pos > 0 else len(text)


def check_cot_state_drift(
    text: str,
    start_pegs: List[List[int]],
    moves: List[List[int]],
    n_disks: int,
) -> bool:
    """True if model's claimed intermediate states match simulation (no drift)."""
    claimed_states = _INTERMEDIATE_STATE_RE.findall(text)
    if not claimed_states:
        return True  # Can't assess → assume OK

    # Simulate and collect actual states after each move
    pegs = pegs_copy(start_pegs)
    actual_states_after: List[List[List[int]]] = []
    for move in moves:
        status, _ = simulate_single_move(pegs, move, n_disks)
        if status != "OK":
            break
        actual_states_after.append(pegs_copy(pegs))

    if not actual_states_after:
        return False

    # Try to match each claimed state to the nearest actual state
    matches = 0
    for cg0, cg1, cg2 in claimed_states:
        try:
            p0 = [int(x.strip()) for x in cg0.split(',') if x.strip().lstrip('-').isdigit()]
            p1 = [int(x.strip()) for x in cg1.split(',') if x.strip().lstrip('-').isdigit()]
            p2 = [int(x.strip()) for x in cg2.split(',') if x.strip().lstrip('-').isdigit()]
            claimed = [p0, p1, p2]
            if any(actual == claimed for actual in actual_states_after):
                matches += 1
        except (ValueError, TypeError):
            continue

    if not claimed_states:
        return True
    return (matches / len(claimed_states)) >= 0.5


# ── Main analysis ─────────────────────────────────────────────────────────────

def analyse(args: argparse.Namespace) -> None:
    with open(args.generated_texts) as f:
        generated_texts: Dict[str, str] = json.load(f)

    n_disks = args.n_disks
    states = enumerate_states(n_disks)
    goal_state: State = tuple([2] * n_disks)
    goal_pegs = state_tuple_to_pegs(goal_state, n_disks)

    print("[INFO] Computing optimal solution lengths…")
    opt_lengths = compute_opt_lengths(states, n_disks)

    # ── Per-problem analysis ──────────────────────────────────────────────────
    records: List[Dict[str, Any]] = []

    for state in states:
        key = str(tuple(int(x) for x in state))
        text = generated_texts.get(key, "")
        start_pegs = state_tuple_to_pegs(state, n_disks)
        opt_len = opt_lengths[state]

        moves = parse_moves(text)

        # ── Simulation & classification ───────────────────────────────────────
        if moves is None:
            sim = {
                "category": PARSE_ERROR,
                "error_step": None,
                "error_fraction": None,
                "legal_prefix_length": 0,
                "state_at_error": None,
                "n_moves": 0,
            }
        else:
            sim = simulate_moves(start_pegs, moves, goal_pegs, n_disks, opt_len)

        category = sim["category"]

        # ── CoT analysis ─────────────────────────────────────────────────────
        cot_chars = cot_length(text)
        has_strat = has_strategic_language(text)
        n_inter = count_intermediate_states(text)
        no_drift = check_cot_state_drift(text, start_pegs, moves or [], n_disks)

        records.append({
            "state": key,
            "state_tuple": list(state),
            "opt_len": opt_len,
            "category": category,
            "error_step": sim["error_step"],
            "error_fraction": sim["error_fraction"],
            "legal_prefix_length": sim["legal_prefix_length"],
            "state_at_error": sim["state_at_error"],
            "n_moves": sim["n_moves"],
            "cot_chars": cot_chars,
            "cot_has_strategy": has_strat,
            "cot_n_intermediate_states": n_inter,
            "cot_no_drift": no_drift,
            "moves_parsed": moves,
        })

    # ── Table 1: Error category summary ──────────────────────────────────────
    counts = Counter(r["category"] for r in records)
    n_total = len(records)

    print()
    print("── Table 1: Error category summary ──────────────────────────────")
    print(f"  {'Category':<28} {'Count':>6}  {'% Total':>8}")
    print(f"  {'-'*28} {'-'*6}  {'-'*8}")
    for cat in ALL_CATEGORIES:
        c = counts.get(cat, 0)
        pct = 100.0 * c / n_total
        print(f"  {cat:<28} {c:>6}  {pct:>7.1f}%")
    print(f"  {'TOTAL':<28} {n_total:>6}  {'100.0%':>8}")

    # ── Table 2: Error location ───────────────────────────────────────────────
    failed = [r for r in records if r["category"] not in CORRECT_CATEGORIES
              and r["category"] != PARSE_ERROR]
    error_steps = [r["error_step"] for r in failed if r["error_step"] is not None]
    error_fracs = [r["error_fraction"] for r in failed if r["error_fraction"] is not None]
    legal_pfxs = [r["legal_prefix_length"] for r in records
                  if r["category"] not in CORRECT_CATEGORIES]

    def safe_mean(xs):
        return sum(xs) / len(xs) if xs else float("nan")

    def safe_median(xs):
        if not xs:
            return float("nan")
        s = sorted(xs)
        mid = len(s) // 2
        return (s[mid] + s[~mid]) / 2

    print()
    print("── Table 2: Error location ───────────────────────────────────────")
    print(f"  {'Metric':<32} {'Value':>8}")
    print(f"  {'-'*32} {'-'*8}")
    print(f"  {'Mean error step (non-parse fails)':<32} {safe_mean(error_steps):>8.2f}")
    print(f"  {'Mean error fraction':<32} {safe_mean(error_fracs):>8.2f}")
    print(f"  {'Mean legal prefix length (all fails)':<32} {safe_mean(legal_pfxs):>8.2f}")
    print(f"  {'Median legal prefix length':<32} {safe_median(legal_pfxs):>8.2f}")

    # ── Table 3: Errors by optimal solution length ────────────────────────────
    buckets = [(1, 3), (4, 7), (8, 11), (12, 15)]
    print()
    print("── Table 3: Errors by optimal solution length ────────────────────")
    print(f"  {'Opt Length':<12} {'Total':>6} {'Correct':>8} {'Error%':>8} {'Most Common Error'}")
    print(f"  {'-'*12} {'-'*6} {'-'*8} {'-'*8} {'-'*25}")
    for lo, hi in buckets:
        bucket_recs = [r for r in records if lo <= r["opt_len"] <= hi]
        correct_recs = [r for r in bucket_recs if r["category"] in CORRECT_CATEGORIES]
        error_recs = [r for r in bucket_recs if r["category"] not in CORRECT_CATEGORIES]
        err_pct = 100.0 * len(error_recs) / len(bucket_recs) if bucket_recs else 0.0
        most_common = Counter(r["category"] for r in error_recs).most_common(1)
        mc_str = most_common[0][0] if most_common else "—"
        print(f"  {f'{lo}-{hi}':<12} {len(bucket_recs):>6} {len(correct_recs):>8} "
              f"{err_pct:>7.1f}% {mc_str}")

    # ── Table 4: CoT analysis ─────────────────────────────────────────────────
    correct_recs = [r for r in records if r["category"] in CORRECT_CATEGORIES]
    failed_recs  = [r for r in records if r["category"] not in CORRECT_CATEGORIES]

    def pct_true(recs, key):
        if not recs:
            return float("nan")
        return 100.0 * sum(bool(r[key]) for r in recs) / len(recs)

    print()
    print("── Table 4: CoT analysis ─────────────────────────────────────────")
    print(f"  {'Metric':<38} {'Correct':>10} {'Failed':>10}")
    print(f"  {'-'*38} {'-'*10} {'-'*10}")
    metrics = [
        ("Mean CoT length (chars)", "cot_chars", False),
        ("% with strategic language", "cot_has_strategy", True),
        ("% with intermediate state tracking", "cot_n_intermediate_states", True),
        ("% where CoT states match sim", "cot_no_drift", True),
    ]
    for label, key, is_pct in metrics:
        if is_pct:
            if key == "cot_n_intermediate_states":
                cv = 100.0 * sum(1 for r in correct_recs if r[key] > 0) / len(correct_recs) if correct_recs else float("nan")
                fv = 100.0 * sum(1 for r in failed_recs if r[key] > 0) / len(failed_recs) if failed_recs else float("nan")
                print(f"  {label:<38} {cv:>9.1f}% {fv:>9.1f}%")
            else:
                cv = pct_true(correct_recs, key)
                fv = pct_true(failed_recs, key)
                print(f"  {label:<38} {cv:>9.1f}% {fv:>9.1f}%")
        else:
            cv = safe_mean([r[key] for r in correct_recs])
            fv = safe_mean([r[key] for r in failed_recs])
            print(f"  {label:<38} {cv:>10,.0f} {fv:>10,.0f}")

    # ── Table 5: Per-problem detail ───────────────────────────────────────────
    print()
    print("── Table 5: Per-problem detail ───────────────────────────────────")
    hdr = f"  {'State':<8} {'OptLen':>6} {'Category':<26} {'ErrStep':>8} {'LegalPfx':>9} {'Strategy':>9}"
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for r in records:
        state_str = "".join(str(x) for x in r["state_tuple"])
        err_s = str(r["error_step"]) if r["error_step"] is not None else "-"
        strat = "Yes" if r["cot_has_strategy"] else "No"
        print(f"  {state_str:<8} {r['opt_len']:>6} {r['category']:<26} "
              f"{err_s:>8} {r['legal_prefix_length']:>9} {strat:>9}")

    # ── Per-category examples ─────────────────────────────────────────────────
    error_cats = [cat for cat in ALL_CATEGORIES
                  if cat not in CORRECT_CATEGORIES and counts.get(cat, 0) > 0]
    print()
    print("── Examples per error category (up to 3 each) ────────────────────")
    by_cat: Dict[str, List[Dict]] = defaultdict(list)
    for r in records:
        by_cat[r["category"]].append(r)

    for cat in error_cats:
        examples = by_cat[cat][:3]
        print(f"\n  [{cat}] ({counts[cat]} total)")
        for ex in examples:
            state_str = "".join(str(x) for x in ex["state_tuple"])
            start_pegs = state_tuple_to_pegs(tuple(ex["state_tuple"]), n_disks)
            print(f"    State {state_str}  (pegs={start_pegs})  opt_len={ex['opt_len']}")
            if ex["moves_parsed"]:
                moves_display = ex["moves_parsed"][:6]
                suffix = "…" if len(ex["moves_parsed"]) > 6 else ""
                print(f"    Moves: {moves_display}{suffix}  ({ex['n_moves']} total)")
            else:
                print("    Moves: <none parsed>")
            if ex["error_step"] is not None:
                print(f"    First error at step {ex['error_step']}, "
                      f"after {ex['legal_prefix_length']} legal moves")
                print(f"    State at error: {ex['state_at_error']}")
            elif cat in {WRONG_GOAL_STATE, PREMATURE_STOP, EXCESSIVE_MOVES}:
                start_ps = state_tuple_to_pegs(tuple(ex["state_tuple"]), n_disks)
                # Simulate to show final state
                pegs = pegs_copy(start_ps)
                for mv in (ex["moves_parsed"] or []):
                    simulate_single_move(pegs, mv, n_disks)
                print(f"    Final state after {ex['n_moves']} moves: {pegs}")
                print(f"    Goal was: {goal_pegs}")

    # ── Save JSON ─────────────────────────────────────────────────────────────
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Strip verbose moves from JSON to keep it tidy (keep first 20)
    save_records = []
    for r in records:
        rec = {k: v for k, v in r.items() if k != "moves_parsed"}
        if r["moves_parsed"] is not None:
            rec["moves_parsed_first20"] = r["moves_parsed"][:20]
            rec["moves_total"] = len(r["moves_parsed"])
        else:
            rec["moves_parsed_first20"] = None
            rec["moves_total"] = 0
        save_records.append(rec)

    summary = {
        "total": n_total,
        "category_counts": dict(counts),
        "records": save_records,
    }
    out_path = out_dir / "evaluate_results.json"
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\n[INFO] Full results saved to {out_path}")


def main() -> None:
    args = parse_args()
    analyse(args)


if __name__ == "__main__":
    main()
