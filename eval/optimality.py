"""Map a generated move sequence into one of {Optimal, Suboptimal, Incorrect, Illegal}.

The training template emits moves as 3-tuples `[disk, from, to]` (1-indexed disk,
matching planning.py and the system prompt). We tolerate 2-tuples too — for
older runs or for robustness against parse drift — by inferring the disk from
the topmost source-peg disk.

Categories returned (string):
- "Optimal"     — reached s_G in optimal number of moves
- "Suboptimal"  — reached s_G but with more moves than BFS-optimal
- "Incorrect"   — every move was legal but the final state != s_G
- "Illegal"     — at least one move was illegal (incl. parse failure)
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Optional, Tuple

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from hanoi_data.template import N_DISKS, state_to_pegs  # noqa: E402
from planning import TowersOfHanoiSolver  # noqa: E402

State = Tuple[int, int, int, int]
Move2 = Tuple[int, int]


def _topmost_planning_disk(peg_list: List[int]) -> Optional[int]:
    return peg_list[-1] if peg_list else None


def simulate_moves(
    s_I: State,
    moves: List[List[int]],
) -> Tuple[List[List[int]], List[List[int]], Optional[State], bool]:
    """Apply a list of moves against the planning-format state. Accepts both
    3-tuples `[disk, from, to]` (1-indexed disk) and legacy 2-tuples
    `[from, to]` (disk inferred from the topmost source-peg disk).

    Returns:
      planning_moves    list of [disk, from, to] (1-indexed disk) up to the first illegal move
      final_pegs        final 3-peg list (planning format)
      final_state       spec tuple, or None if final state is invalid
      had_illegal       True if any move was rejected
    """
    pegs = [list(p) for p in state_to_pegs(tuple(int(x) for x in s_I))]
    planning_moves: List[List[int]] = []
    had_illegal = False
    for m in moves:
        if not isinstance(m, list):
            had_illegal = True
            break
        if len(m) == 3:
            declared_disk, fp, tp = int(m[0]), int(m[1]), int(m[2])
        elif len(m) == 2:
            declared_disk = None
            fp, tp = int(m[0]), int(m[1])
        else:
            had_illegal = True
            break
        if not (0 <= fp <= 2 and 0 <= tp <= 2) or fp == tp:
            had_illegal = True
            break
        actual_disk = _topmost_planning_disk(pegs[fp])
        if actual_disk is None:
            had_illegal = True
            break
        if declared_disk is not None and declared_disk != actual_disk:
            # Disk claim disagrees with reality → illegal.
            had_illegal = True
            break
        if pegs[tp] and pegs[tp][-1] < actual_disk:
            had_illegal = True
            break
        pegs[fp].pop()
        pegs[tp].append(actual_disk)
        planning_moves.append([actual_disk, fp, tp])

    # Convert pegs back to spec tuple for clarity.
    spec = [0] * N_DISKS
    for p, peg_list in enumerate(pegs):
        for d in peg_list:
            spec[d - 1] = p
    return planning_moves, pegs, tuple(spec), had_illegal


# Backwards-compat alias for older callers.
simulate_two_tuple_moves = simulate_moves


def classify_moves(
    s_I: State,
    s_G: State,
    moves: Optional[List[List[int]]],
) -> dict:
    """Return {category, n_moves, optimal_len, final_state, had_illegal}.

    `moves` may be None when extraction failed entirely → "Illegal".
    """
    s_I_t = tuple(int(x) for x in s_I)
    s_G_t = tuple(int(x) for x in s_G)
    goal_pegs = state_to_pegs(s_G_t)

    solver = TowersOfHanoiSolver()
    optimal_path = solver.solve(
        num_disks=N_DISKS,
        initial_state=state_to_pegs(s_I_t),
        goal_state=goal_pegs,
    )
    optimal_len = len(optimal_path) if optimal_path else 0

    if moves is None:
        return {
            "category": "Illegal",
            "n_moves": 0,
            "optimal_len": optimal_len,
            "final_state": list(s_I_t),
            "had_illegal": True,
        }

    planning_moves, final_pegs, final_state, had_illegal = simulate_moves(
        s_I_t, moves
    )

    solved = final_pegs == goal_pegs
    if had_illegal:
        category = "Illegal"
    elif not solved:
        category = "Incorrect"
    elif len(planning_moves) == optimal_len:
        category = "Optimal"
    else:
        category = "Suboptimal"

    return {
        "category": category,
        "n_moves": len(planning_moves),
        "optimal_len": optimal_len,
        "final_state": list(final_state) if final_state is not None else list(s_I_t),
        "had_illegal": had_illegal,
    }
