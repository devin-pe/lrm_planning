#!/usr/bin/env python3
"""Deterministic move-replay ground truth for ToH reasoning traces.

For each generated trace we extract the model's COMMITTED move list and replay it
from the initial state to produce the true state trajectory - no LLM needed. This
is the reference a state tracker should be scored against.

Handling the variance found by surveying the traces:
- Move format (both qwen & deepseek): `moves = [[disk, from, to], ...]`.
- Disk ids are 1..4 (1 = smallest) in the trace; our state tuples index disks 0..3,
  so disk_id is mapped d -> d-1 on replay.
- Pegs are 0..2, order is [disk, from, to] for both models.
- Primary extraction: the LAST complete bracket-matched `moves = [...]` block
  (planning._extract_moves_block) parsed as JSON (tolerant to trailing commas / #
  comments). This is the committed answer and is most reliable for goal-reaching.
- Fallback (truncated / dirty blocks, e.g. answers cut at the 32k token cap):
  regex each [d,f,t] triple in that block in order and replay until the first
  ILLEGAL move (recovers the legal prefix).
- Legality is enforced during replay; illegal committed plans (common for
  deepseek) yield only their legal prefix, flagged as not goal-reaching.

Output: outputs/tracker_ground_truth/ground_truth.json
  { "<source>|<problem_id>": {
        "source", "s_I", "s_G", "committed_moves" (0-indexed disks),
        "trajectory" [[..],...],  "n_moves", "reaches_goal",
        "full_block_legal" (whole extracted block replayed legally),
        "extraction" ("strict" | "prefix") } , ... }
"""

from __future__ import annotations

import ast
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from planning import _extract_moves_block, _parse_moves_json

State = Tuple[int, int, int, int]
GOAL: State = (2, 2, 2, 2)
MOVE_RE = re.compile(r'\[\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\]')

SOURCES = [
    ("outputs/qwen_probe/generated_texts.json", "qwen"),
    ("outputs/deepseek_r1_qwen32b_probe/generated_texts.json", "deepseek"),
]


def apply_move(st: State, disk_id: int, fr: int, to: int) -> Optional[State]:
    """Apply one move [disk_id(1..4), from, to]; return new state or None if illegal."""
    d = disk_id - 1
    if not (0 <= d <= 3 and 0 <= fr <= 2 and 0 <= to <= 2):
        return None
    if st[d] != fr:
        return None
    if any(st[o] == fr and o < d for o in range(4)):   # a smaller disk covers it
        return None
    if any(st[o] == to and o < d for o in range(4)):   # would sit on a smaller disk
        return None
    ns = list(st)
    ns[d] = to
    return tuple(ns)


def replay(s_I: State, moves: List[List[int]]) -> Tuple[List[State], int, bool]:
    """Replay moves until the first illegal one. Returns (trajectory, n_legal, full_legal)."""
    st = tuple(s_I)
    traj = [st]
    for i, m in enumerate(moves):
        if len(m) != 3:
            return traj, len(traj) - 1, False
        nx = apply_move(st, m[0], m[1], m[2])
        if nx is None:
            return traj, len(traj) - 1, False
        st = nx
        traj.append(st)
    return traj, len(traj) - 1, True


def committed_moves(text: str) -> Tuple[List[List[int]], str]:
    """Extract the committed move list. Returns (moves, extraction_mode)."""
    block = _extract_moves_block(text)
    if block:
        try:
            moves = _parse_moves_json(block)
            if isinstance(moves, list) and all(isinstance(m, list) for m in moves):
                return moves, "strict"
        except Exception:
            pass
        # Fallback: regex triples out of the (possibly truncated/dirty) block.
        moves = [[int(a), int(b), int(c)] for a, b, c in MOVE_RE.findall(block)]
        if moves:
            return moves, "prefix"
    return [], "none"


def ground_truth_for_trace(s_I: State, text: str) -> Dict:
    moves, mode = committed_moves(text)
    traj, n_legal, full_legal = replay(s_I, moves)
    legal_moves_0idx = [[m[0] - 1, m[1], m[2]] for m in moves[:n_legal]]
    return {
        "s_I": list(s_I), "s_G": list(GOAL),
        "committed_moves": legal_moves_0idx,
        "trajectory": [list(s) for s in traj],
        "n_moves": n_legal,
        "reaches_goal": traj[-1] == GOAL,
        "full_block_legal": full_legal,
        "extraction": mode,
    }


def main() -> None:
    out_dir = Path("outputs/tracker_ground_truth")
    out_dir.mkdir(parents=True, exist_ok=True)
    gt: Dict[str, Dict] = {}
    summary = {}
    for path, source in SOURCES:
        g = json.loads(Path(path).read_text(encoding="utf-8"))
        n = ngoal = nfulllegal = nnone = 0
        mlens = []
        for k, text in g.items():
            s_I = tuple(ast.literal_eval(k))
            rec = ground_truth_for_trace(s_I, text)
            rec["source"] = source
            rec["problem_id"] = k
            gt[f"{source}|{k}"] = rec
            n += 1
            ngoal += int(rec["reaches_goal"])
            nfulllegal += int(rec["full_block_legal"])
            nnone += int(rec["extraction"] == "none")
            mlens.append(rec["n_moves"])
        import statistics
        summary[source] = {"n": n, "reaches_goal": ngoal, "full_block_legal": nfulllegal,
                           "no_block": nnone,
                           "median_committed_moves": int(statistics.median(mlens)) if mlens else 0}
    (out_dir / "ground_truth.json").write_text(json.dumps(gt, indent=2), encoding="utf-8")
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("Move-replay ground-truth coverage:")
    print(f"{'source':10} | {'n':>3} | {'goal-reaching':>13} | {'full-block-legal':>16} | {'no-block':>8} | {'median moves':>12}")
    print("-" * 78)
    for s, v in summary.items():
        print(f"{s:10} | {v['n']:>3} | {v['reaches_goal']:>13} | {v['full_block_legal']:>16} | "
              f"{v['no_block']:>8} | {v['median_committed_moves']:>12}")
    total_goal = sum(v["reaches_goal"] for v in summary.values())
    print(f"\nUsable clean ground truth (legal committed solution reaching goal): "
          f"{total_goal} traces")
    print(f"Wrote {out_dir}/ground_truth.json  (+ summary.json)")


if __name__ == "__main__":
    main()
