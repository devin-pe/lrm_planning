"""Templated reasoning-trace generator for 4-disk Tower of Hanoi.

Convention reconciliation with planning.py
==========================================
- planning.py uses 1-indexed disks (1 = smallest, N = largest) stored as a
  list of 3 pegs, each peg a list of disk numbers bottom-to-top
  (so pegs[p][-1] is the smallest / movable disk on peg p).
- This module exposes the spec convention: a state is a tuple
  (p_0, p_1, p_2, p_3) where p_i is the peg of disk i (0-indexed,
  0 = smallest, 3 = largest).

The two helpers `state_to_pegs` / `pegs_to_state` convert between the two
formats. They are the only place either convention is mixed; everything else
in this module works in spec convention.
"""

from __future__ import annotations

import itertools
import sys
from pathlib import Path
from typing import List, Optional, Tuple

# Allow running as a script (`python hanoi_data/template.py`) by adding the
# repo root to sys.path so the top-level planning module imports.
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from planning import TowersOfHanoiSolver, TowersOfHanoiState
from prompts import create_nonstandard_prompt

State = Tuple[int, ...]
N_DISKS = 4


# ── Convention conversion ──────────────────────────────────────────────────────

def state_to_pegs(state: State) -> List[List[int]]:
    """Spec tuple → planning-pegs list (largest-first per peg)."""
    pegs: List[List[int]] = [[], [], []]
    # Place largest disk first so it sits on the bottom (planning is bottom→top).
    for spec_disk in range(len(state) - 1, -1, -1):
        pegs[state[spec_disk]].append(spec_disk + 1)
    return pegs


def pegs_to_state(pegs: List[List[int]], n_disks: int = N_DISKS) -> State:
    """Planning-pegs list → spec tuple."""
    out = [0] * n_disks
    for p, peg_list in enumerate(pegs):
        for planning_disk in peg_list:
            out[planning_disk - 1] = p
    return tuple(out)


# ── Required public API (spec convention) ────────────────────────────────────

def enumerate_states(n_disks: int = N_DISKS) -> List[State]:
    """All 3^n valid states (every assignment of disks to pegs is reachable)."""
    return [tuple(s) for s in itertools.product((0, 1, 2), repeat=n_disks)]


def topmost_disk(state: State, peg: int) -> Optional[int]:
    """Smallest disk on `peg` (the only movable one), or None if peg is empty."""
    for d in range(len(state)):
        if state[d] == peg:
            return d
    return None


def legal_moves(state: State) -> List[Tuple[int, int]]:
    """All legal (from_peg, to_peg) moves from `state`."""
    pegs = state_to_pegs(state)
    s = TowersOfHanoiState(num_disks=len(state), pegs=pegs)
    moves: List[Tuple[int, int]] = []
    for fp in range(3):
        for tp in range(3):
            if fp == tp:
                continue
            if s.is_valid_move(fp, tp):
                moves.append((fp, tp))
    return moves


def apply_move(state: State, from_peg: int, to_peg: int) -> State:
    """Apply a legal move; raise ValueError if illegal."""
    pegs = state_to_pegs(state)
    s = TowersOfHanoiState(num_disks=len(state), pegs=pegs)
    if not s.apply_move(from_peg, to_peg):
        raise ValueError(
            f"Illegal move ({from_peg}, {to_peg}) from state {state}"
        )
    return pegs_to_state(s.pegs, n_disks=len(state))


def bfs_optimal(s_I: State, s_G: State) -> List[Tuple[int, int, int]]:
    """Wrap planning's BFS solver. Returns 3-tuples (disk, from_peg, to_peg)
    with disks 1-indexed (matches prompts.py's `moves = [[disk, from, to], …]`
    convention).
    """
    solver = TowersOfHanoiSolver()
    moves = solver.solve(
        num_disks=len(s_I),
        initial_state=state_to_pegs(s_I),
        goal_state=state_to_pegs(s_G),
    )
    if moves is None:
        raise RuntimeError(f"No path found between {s_I} and {s_G}")
    return [(int(m[0]), int(m[1]), int(m[2])) for m in moves]


# ── Trace generation ─────────────────────────────────────────────────────────

def get_prompts(s_I: State, s_G: State) -> Tuple[str, str]:
    """Return (system_prompt, user_prompt) for one (s_I, s_G) pair, using
    the canonical Tower-of-Hanoi prompts defined in prompts.py.

    The chat template is NOT applied here — apply it at tokenization time
    with the actual tokenizer (`tokenizer.apply_chat_template`).
    """
    s_I_t = tuple(int(x) for x in s_I)
    s_G_t = tuple(int(x) for x in s_G)
    sys_p, user_p, _ = create_nonstandard_prompt(
        num_disks=len(s_I_t),
        problem_id=0,
        seed=0,
        initial_state_override=state_to_pegs(s_I_t),
        goal_state_override=state_to_pegs(s_G_t),
    )
    return sys_p, user_p


def _state_repr(s: State) -> str:
    """Match the Python list repr used in the prompt examples: '[0, 1, 2, 2]'."""
    return str(list(s))


def generate_trace(
    s_I: State,
    s_G: State,
    moves: List[Tuple[int, int, int]],
) -> dict:
    """Build the (prompt, completion, supervisions) record for one problem.

    Supervisions are character spans of the state literals inside the
    'State: {before} -> {after}.' lines only — not the framing 'Starting
    from …' line and not the final 'moves = [...]' block.
    """
    s_I_t = tuple(int(x) for x in s_I)
    s_G_t = tuple(int(x) for x in s_G)
    if s_I_t == s_G_t:
        raise ValueError("s_I == s_G; spec excludes the no-op case")
    if not moves:
        raise ValueError("Empty move list but s_I != s_G")

    system_prompt, user_prompt = get_prompts(s_I_t, s_G_t)

    completion_parts: List[str] = []
    supervisions: List[Tuple[int, int, List[int]]] = []
    pos = 0  # mirrors len("".join(completion_parts)) cheaply

    def append(s: str) -> None:
        nonlocal pos
        completion_parts.append(s)
        pos += len(s)

    def append_supervised(state: State) -> None:
        nonlocal pos
        literal = _state_repr(state)
        start = pos
        completion_parts.append(literal)
        pos += len(literal)
        supervisions.append((start, pos, list(state)))

    append("<think>\n")
    append(f"Starting from {_state_repr(s_I_t)}, I need to reach "
           f"{_state_repr(s_G_t)} in {len(moves)} moves.\n\n")

    cur = s_I_t
    for i, mv in enumerate(moves, start=1):
        if len(mv) != 3:
            raise ValueError(f"Move {i} must be a 3-tuple (disk, from, to), got {mv}")
        disk_1, fp, tp = int(mv[0]), int(mv[1]), int(mv[2])
        # Cross-check the stated disk against the topmost disk on the source
        # peg. `topmost_disk` returns the spec-style 0-indexed disk (smallest
        # disk on the peg). We display 1-indexed disks to match the prompt's
        # "disks are numbered from 1 (smallest) to N (largest)" wording.
        disk_0 = topmost_disk(cur, fp)
        if disk_0 is None:
            raise ValueError(
                f"Move {i} = ({disk_1}, {fp}, {tp}) is illegal: peg {fp} empty in {cur}"
            )
        if disk_0 + 1 != disk_1:
            raise ValueError(
                f"Move {i} claims disk {disk_1} on peg {fp}, but topmost is "
                f"disk {disk_0 + 1} (state {cur})"
            )
        nxt = apply_move(cur, fp, tp)
        justification = (
            f"This places disk {disk_1} on its goal peg."
            if nxt[disk_0] == s_G_t[disk_0]
            else "This sets up the next move."
        )

        append(
            f"Move {i}: disk {disk_1} is on peg {fp}, moving to peg {tp}. "
            f"{justification}\n"
        )
        append("State: ")
        append_supervised(cur)
        append(" -> ")
        append_supervised(nxt)
        append(".\n\n")

        cur = nxt

    if cur != s_G_t:
        raise ValueError(
            f"Move sequence does not reach s_G: ended at {cur}, expected {s_G_t}"
        )

    append("All disks are in their goal positions.\n</think>\n\n")
    append(f"moves = {[list(m) for m in moves]}\n")

    completion = "".join(completion_parts)

    return {
        "s_I": list(s_I_t),
        "s_G": list(s_G_t),
        "moves": [list(m) for m in moves],
        "system_prompt": system_prompt,
        "user_prompt": user_prompt,
        "completion": completion,
        "supervisions": [[cs, ce, list(t)] for cs, ce, t in supervisions],
    }
