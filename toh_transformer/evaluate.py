#!/usr/bin/env python3
"""Evaluate ToHTransformer on all start/goal pairs with error categorization.

For n_disks=4, this evaluates all 81x81 = 6561 ordered (start, goal) problems,
including trivial start==goal cases.
"""

from __future__ import annotations

import argparse
import itertools
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

if __package__ is None or __package__ == "":
    import sys

    sys.path.append(str(Path(__file__).resolve().parent.parent))

from planning import TowersOfHanoiSolver
from toh_transformer.config import default_model_hparams, max_seq_len_for_disks
from toh_transformer.data import Vocabulary
from toh_transformer.model import ToHTransformer

State = Tuple[int, ...]
Move = Tuple[int, int]

EXPECTED_TOKEN_MAPPING = {
    "P0": 0,
    "P1": 1,
    "P2": 2,
    "M01": 3,
    "M02": 4,
    "M10": 5,
    "M12": 6,
    "M20": 7,
    "M21": 8,
    "BOS": 9,
    "SEP": 10,
    "EOS": 11,
    "PAD": 12,
}

CATEGORY_ORDER = [
    "CORRECT_OPTIMAL",
    "CORRECT_SUBOPTIMAL",
    "WRONG_GOAL_STATE",
    "ILLEGAL_EMPTY_SOURCE",
    "ILLEGAL_LARGER_ON_SMALLER",
    "ILLEGAL_INVALID_TOKEN",
    "PREMATURE_EOS",
    "NO_EOS",
    "TRIVIAL_WRONG",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate ToHTransformer on all start/goal pairs")
    parser.add_argument("--checkpoint", type=str, default="best.pt", help="Path to model checkpoint")
    parser.add_argument("--n_disks", type=int, default=4, help="Number of disks (default: 4)")
    parser.add_argument("--output_dir", type=str, default="toh_transformer/eval_output", help="Output directory")
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device (default: cuda if available else cpu)",
    )
    return parser.parse_args()


def confirm_tokenizer_mapping(vocab: Vocabulary) -> None:
    if vocab.stoi != EXPECTED_TOKEN_MAPPING:
        raise ValueError(
            "Tokenizer mapping mismatch. "
            f"Expected={EXPECTED_TOKEN_MAPPING}, actual={vocab.stoi}"
        )


def enumerate_states(n_disks: int) -> List[State]:
    return [tuple(s) for s in itertools.product((0, 1, 2), repeat=n_disks)]


def tuple_to_pegs(state_tuple: State, n_disks: int) -> List[List[int]]:
    """Convert tuple state to peg lists storing disk ids bottom->top.

    tuple index 0 is smallest disk, so disk_id = idx+1.
    """
    pegs = [[], [], []]
    for disk_idx in range(n_disks - 1, -1, -1):
        disk_id = disk_idx + 1
        pegs[state_tuple[disk_idx]].append(disk_id)
    return [list(p) for p in pegs]


def state_from_pegs(pegs: Sequence[Sequence[int]], n_disks: int) -> State:
    out = [0] * n_disks
    for peg_id, peg in enumerate(pegs):
        for disk_id in peg:
            out[disk_id - 1] = peg_id
    return tuple(out)


def build_context_ids(start: State, goal: State, vocab: Vocabulary) -> List[int]:
    return [
        vocab.bos_id,
        *[vocab.stoi[f"P{p}"] for p in start],
        vocab.sep_id,
        *[vocab.stoi[f"P{p}"] for p in goal],
        vocab.sep_id,
    ]


def resolve_checkpoint_path(path_str: str, n_disks: int) -> Path:
    p = Path(path_str)
    if p.exists():
        return p

    # Helpful fallback for default invocations from repo root.
    if path_str == "best.pt":
        alt = Path(f"toh_transformer/checkpoints/n{n_disks}/best.pt")
        if alt.exists():
            print(f"[INFO] Checkpoint 'best.pt' not found; using {alt}")
            return alt

    raise FileNotFoundError(f"Checkpoint not found: {path_str}")


def load_model(checkpoint_path: Path, n_disks: int, device: torch.device) -> ToHTransformer:
    ckpt = torch.load(checkpoint_path, map_location=device)
    cfg = ckpt.get("config", {}) if isinstance(ckpt, dict) else {}

    defaults = default_model_hparams(n_disks)
    n_layers = int(cfg.get("n_layers", defaults["n_layers"]))
    n_heads = int(cfg.get("n_heads", defaults["n_heads"]))
    d_model = int(cfg.get("d_model", defaults["d_model"]))
    d_ff = int(cfg.get("d_ff", defaults["d_ff"]))
    dropout = float(cfg.get("dropout", defaults["dropout"]))
    max_seq_len = int(cfg.get("max_seq_len", max_seq_len_for_disks(n_disks)))

    model = ToHTransformer(
        vocab_size=len(Vocabulary()),
        max_seq_len=max_seq_len,
        n_layers=n_layers,
        n_heads=n_heads,
        d_model=d_model,
        d_ff=d_ff,
        dropout=dropout,
    ).to(device)

    if isinstance(ckpt, dict) and "model_state" in ckpt:
        state_dict = ckpt["model_state"]
    elif isinstance(ckpt, dict):
        state_dict = ckpt
    else:
        raise ValueError("Checkpoint format not recognized")

    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model


@torch.no_grad()
def greedy_decode(
    model: ToHTransformer,
    context_ids: Sequence[int],
    max_seq_len: int,
    eos_id: int,
    device: torch.device,
) -> Tuple[List[int], bool]:
    """Return generated token ids after context and whether EOS appeared."""
    seq = torch.tensor(context_ids, dtype=torch.long, device=device).unsqueeze(0)
    generated: List[int] = []

    while seq.size(1) < max_seq_len:
        logits = model(seq)
        next_id = int(torch.argmax(logits[:, -1, :], dim=-1).item())
        generated.append(next_id)
        next_token = torch.tensor([[next_id]], dtype=torch.long, device=device)
        seq = torch.cat([seq, next_token], dim=1)

        if next_id == eos_id:
            return generated, True

    return generated, False


def build_optimal_cache(n_disks: int, states: Sequence[State]) -> Dict[Tuple[State, State], Tuple[Move, ...]]:
    solver = TowersOfHanoiSolver()
    cache: Dict[Tuple[State, State], Tuple[Move, ...]] = {}

    for start in states:
        start_pegs = tuple_to_pegs(start, n_disks)
        for goal in states:
            goal_pegs = tuple_to_pegs(goal, n_disks)
            solution = solver.solve(num_disks=n_disks, initial_state=start_pegs, goal_state=goal_pegs)
            if solution is None:
                raise RuntimeError(f"No solution from {start} to {goal}")
            cache[(start, goal)] = tuple((int(m[1]), int(m[2])) for m in solution)

    return cache


def decode_move_token(tok: str) -> Optional[Move]:
    if len(tok) != 3 or tok[0] != "M":
        return None
    if tok[1] not in "012" or tok[2] not in "012":
        return None
    return int(tok[1]), int(tok[2])


def simulate_and_classify(
    start: State,
    goal: State,
    generated_ids: Sequence[int],
    eos_seen: bool,
    vocab: Vocabulary,
    optimal_moves: Sequence[Move],
    n_disks: int,
) -> Dict[str, object]:
    """Classify prediction into exactly one required category."""
    generated_tokens = [vocab.itos[i] for i in generated_ids]

    # Predicted move sequence: between second SEP and EOS, exclusive.
    if eos_seen:
        eos_pos = generated_ids.index(vocab.eos_id)
        move_tokens = generated_tokens[:eos_pos]
    else:
        move_tokens = generated_tokens

    predicted_moves: List[str] = move_tokens
    optimal_move_tokens = [f"M{a}{b}" for (a, b) in optimal_moves]

    result: Dict[str, object] = {
        "start": list(start),
        "goal": list(goal),
        "predicted_moves": predicted_moves,
        "optimal_moves": optimal_move_tokens,
        "optimal_length": len(optimal_moves),
        "predicted_length": len(predicted_moves),
    }

    # Trivial tasks should terminate immediately.
    if start == goal:
        first_is_eos = len(generated_ids) > 0 and generated_ids[0] == vocab.eos_id
        if first_is_eos:
            result["category"] = "CORRECT_OPTIMAL"
            return result

        result["category"] = "TRIVIAL_WRONG"
        result["error_step"] = 1
        if not generated_ids:
            result["error_detail"] = "No token was generated"
        else:
            result["error_detail"] = f"Expected EOS immediately, got {generated_tokens[0]}"
        return result

    # Simulate predicted moves and identify first illegal step.
    pegs = tuple_to_pegs(start, n_disks)
    first_illegal_category: Optional[str] = None
    first_illegal_step: Optional[int] = None
    first_illegal_detail: Optional[str] = None

    for step, tok in enumerate(predicted_moves, start=1):
        move = decode_move_token(tok)
        if move is None:
            first_illegal_category = "ILLEGAL_INVALID_TOKEN"
            first_illegal_step = step
            first_illegal_detail = f"Encountered non-move token {tok}"
            break

        src, dst = move
        if len(pegs[src]) == 0:
            first_illegal_category = "ILLEGAL_EMPTY_SOURCE"
            first_illegal_step = step
            first_illegal_detail = f"Tried to move from empty peg {src}"
            break

        moving = pegs[src][-1]
        if len(pegs[dst]) > 0 and pegs[dst][-1] < moving:
            first_illegal_category = "ILLEGAL_LARGER_ON_SMALLER"
            first_illegal_step = step
            first_illegal_detail = (
                f"Tried to place disk {moving} on smaller disk {pegs[dst][-1]} (peg {dst})"
            )
            break

        pegs[src].pop()
        pegs[dst].append(moving)

    if first_illegal_category is not None:
        result["category"] = first_illegal_category
        result["error_step"] = first_illegal_step
        result["error_detail"] = first_illegal_detail
        return result

    final_state = state_from_pegs(pegs, n_disks)
    reached_goal = final_state == goal

    if eos_seen and not reached_goal:
        result["category"] = "PREMATURE_EOS"
        result["error_step"] = len(predicted_moves) + 1
        result["error_detail"] = "Model emitted EOS before reaching goal state"
        return result

    if not eos_seen:
        result["category"] = "NO_EOS"
        result["error_step"] = len(predicted_moves)
        result["error_detail"] = "Reached max sequence length without EOS"
        return result

    if predicted_moves == optimal_move_tokens:
        result["category"] = "CORRECT_OPTIMAL"
        return result

    if reached_goal and len(predicted_moves) > len(optimal_move_tokens):
        result["category"] = "CORRECT_SUBOPTIMAL"
        return result

    if reached_goal:
        # Should be rare with BFS optimal reference, but keep category assignment total.
        result["category"] = "CORRECT_SUBOPTIMAL"
        result["error_detail"] = "Reached goal with alternative non-identical legal sequence"
        return result

    result["category"] = "WRONG_GOAL_STATE"
    result["error_step"] = len(predicted_moves)
    result["error_detail"] = f"All moves legal but ended at state {list(final_state)} instead of goal {list(goal)}"
    return result


def print_summary(results: Sequence[Dict[str, object]]) -> None:
    total = len(results)
    counts = {cat: 0 for cat in CATEGORY_ORDER}
    for row in results:
        counts[str(row["category"])] += 1

    print("\nCategory                  | Count | % of Total")
    print("--------------------------+-------+-----------")
    for cat in CATEGORY_ORDER:
        cnt = counts[cat]
        pct = 100.0 * cnt / total
        print(f"{cat:<26} | {cnt:5d} | {pct:9.2f}")

    total_correct = counts["CORRECT_OPTIMAL"] + counts["CORRECT_SUBOPTIMAL"]
    total_incorrect = total - total_correct
    print("--------------------------+-------+-----------")
    print(f"{'TOTAL CORRECT':<26} | {total_correct:5d} | {100.0 * total_correct / total:9.2f}")
    print(f"{'TOTAL INCORRECT':<26} | {total_incorrect:5d} | {100.0 * total_incorrect / total:9.2f}")


def print_examples(results: Sequence[Dict[str, object]], max_examples: int = 5) -> None:
    by_category: Dict[str, List[Dict[str, object]]] = defaultdict(list)
    for row in results:
        by_category[str(row["category"])].append(row)

    for cat in CATEGORY_ORDER:
        rows = by_category.get(cat, [])
        if not rows or cat.startswith("CORRECT"):
            continue

        print(f"\nExamples for {cat} (showing up to {max_examples}):")
        for ex in rows[:max_examples]:
            start = tuple(ex["start"])  # type: ignore[arg-type]
            goal = tuple(ex["goal"])  # type: ignore[arg-type]
            optimal_len = ex["optimal_length"]
            pred_moves = " ".join(ex["predicted_moves"])  # type: ignore[arg-type]
            error_step = ex.get("error_step")
            detail = ex.get("error_detail", "")

            print(f"  start={start} goal={goal}")
            print(f"  optimal_moves={optimal_len}")
            print(f"  predicted={pred_moves if pred_moves else '<empty>'}")
            if error_step is not None:
                print(f"  error_step={error_step}")
            if detail:
                print(f"  detail={detail}")
            print()


def print_error_by_optimal_length(results: Sequence[Dict[str, object]]) -> None:
    totals = defaultdict(int)
    incorrects = defaultdict(int)

    for row in results:
        opt_len = int(row["optimal_length"])
        totals[opt_len] += 1
        cat = str(row["category"])
        is_incorrect = cat not in {"CORRECT_OPTIMAL", "CORRECT_SUBOPTIMAL"}
        if is_incorrect:
            incorrects[opt_len] += 1

    print("\nIncorrect rate by optimal path length:")
    print("Optimal Length | Total | Incorrect | Error Rate (%)")
    print("--------------+-------+-----------+---------------")
    for opt_len in sorted(totals.keys()):
        t = totals[opt_len]
        inc = incorrects[opt_len]
        rate = 100.0 * inc / max(t, 1)
        print(f"{opt_len:13d} | {t:5d} | {inc:9d} | {rate:13.2f}")


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("Requested cuda but CUDA is not available")

    if args.n_disks != 4:
        print("[WARN] Script targets n_disks=4 by default; running with provided n_disks")

    vocab = Vocabulary()
    confirm_tokenizer_mapping(vocab)

    checkpoint_path = resolve_checkpoint_path(args.checkpoint, args.n_disks)
    model = load_model(checkpoint_path, args.n_disks, device)

    states = enumerate_states(args.n_disks)
    problems = [(s, g) for s in states for g in states]  # includes trivial start==goal

    print(f"[INFO] Loaded checkpoint: {checkpoint_path}")
    print(f"[INFO] Evaluating {len(problems)} problems ({len(states)} states x {len(states)} states)")
    print(f"[INFO] Device: {device}")

    print("[INFO] Computing optimal references...")
    optimal_cache = build_optimal_cache(args.n_disks, states)

    results: List[Dict[str, object]] = []
    for idx, (start, goal) in enumerate(problems, start=1):
        context_ids = build_context_ids(start, goal, vocab)
        generated_ids, eos_seen = greedy_decode(
            model=model,
            context_ids=context_ids,
            max_seq_len=model.max_seq_len,
            eos_id=vocab.eos_id,
            device=device,
        )

        optimal_moves = optimal_cache[(start, goal)]
        row = simulate_and_classify(
            start=start,
            goal=goal,
            generated_ids=generated_ids,
            eos_seen=eos_seen,
            vocab=vocab,
            optimal_moves=optimal_moves,
            n_disks=args.n_disks,
        )
        results.append(row)

        if idx % 500 == 0:
            print(f"[INFO] Processed {idx}/{len(problems)} problems")

    print_summary(results)
    print_examples(results, max_examples=5)
    print_error_by_optimal_length(results)

    out_path = output_dir / "evaluation_results.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\n[INFO] Saved full results to {out_path}")

    # Small machine-readable summary for scripting.
    counts = defaultdict(int)
    for row in results:
        counts[str(row["category"])] += 1
    summary = {
        "n_problems": len(results),
        "counts": {cat: int(counts[cat]) for cat in CATEGORY_ORDER},
        "total_correct": int(counts["CORRECT_OPTIMAL"] + counts["CORRECT_SUBOPTIMAL"]),
        "total_incorrect": int(len(results) - counts["CORRECT_OPTIMAL"] - counts["CORRECT_SUBOPTIMAL"]),
    }
    summary_path = output_dir / "evaluation_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"[INFO] Saved summary to {summary_path}")


if __name__ == "__main__":
    main()
