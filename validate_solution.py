"""
Validate TOH solutions from new_baseline_results problem JSON files.

Usage:
    python validate_solution.py problem_003.json                           # single file
    python validate_solution.py new_baseline_results/run_20260207_122942/  # entire run dir
    python validate_solution.py new_baseline_results/                      # all runs
    python validate_solution.py -v problem_003.json                        # verbose step-by-step
"""

import argparse
import copy
import json
import os
import re
import sys
from collections import deque
from pathlib import Path


def state_to_tuple(state):
    """Convert state (list of lists) to hashable tuple for BFS."""
    return tuple(tuple(peg) for peg in state)


def extract_moves(response_text):
    """
    Extract the moves array from the model's response text.
    Looks for `moves = [[...]]` pattern outside <think> tags.
    """
    text = re.sub(r'<think>.*?</think>', '', response_text, flags=re.DOTALL)
    # Strip inline comments (# ...) before matching — models often add them
    text = re.sub(r'#[^\n]*', '', text)
    pattern = r'moves\s*=\s*(\[(?:\s*\[[^\]]+\]\s*,?\s*)+\])'
    matches = re.findall(pattern, text, re.DOTALL)
    if not matches:
        return None
    moves_str = matches[-1].strip()
    # Clean up any trailing commas before ]
    moves_str = re.sub(r',\s*\]', ']', moves_str)
    try:
        return json.loads(moves_str)
    except json.JSONDecodeError:
        return None


def validate_moves(initial_state, goal_state, moves, num_disks, verbose=False):
    """
    Validate a sequence of moves step by step.

    Returns:
        dict with validation results
    """
    state = copy.deepcopy(initial_state)
    violations = []

    if verbose:
        print(f"  Initial state: {state}")
        print(f"  Goal state:    {goal_state}")
        print(f"  Num moves:     {len(moves)}")
        print("  " + "-" * 56)

    for i, move in enumerate(moves):
        if len(move) != 3:
            violations.append(f"Step {i+1}: Invalid move format {move}")
            if verbose:
                print(f"  Step {i+1}: VIOLATION - bad format {move}")
            continue

        disk, from_peg, to_peg = move

        if disk < 1 or disk > num_disks:
            violations.append(f"Step {i+1}: Invalid disk number {disk}")
            if verbose:
                print(f"  Step {i+1}: VIOLATION - invalid disk {disk}")
            continue

        if from_peg < 0 or from_peg > 2 or to_peg < 0 or to_peg > 2:
            violations.append(f"Step {i+1}: Invalid peg number")
            if verbose:
                print(f"  Step {i+1}: VIOLATION - invalid peg")
            continue

        if not state[from_peg] or state[from_peg][-1] != disk:
            top = state[from_peg][-1] if state[from_peg] else "empty"
            violations.append(
                f"Step {i+1}: Disk {disk} not on top of peg {from_peg} (top={top})"
            )
            if verbose:
                print(f"  Step {i+1}: VIOLATION - disk {disk} not on top of peg {from_peg} (top={top})")
            continue

        if state[to_peg] and state[to_peg][-1] < disk:
            violations.append(
                f"Step {i+1}: Cannot place disk {disk} on smaller disk {state[to_peg][-1]}"
            )
            if verbose:
                print(f"  Step {i+1}: VIOLATION - disk {disk} on smaller disk {state[to_peg][-1]}")
            continue

        state[from_peg].pop()
        state[to_peg].append(disk)
        if verbose:
            print(f"  Step {i+1}: Move disk {disk}: peg {from_peg} -> peg {to_peg}  =>  {state}")

    solved = state == goal_state

    if verbose:
        print("  " + "-" * 56)
        print(f"  Final state: {state}")
        print(f"  Goal state:  {goal_state}")
        print(f"  Solved:      {solved}")
        print(f"  Violations:  {len(violations)}")
        for v in violations:
            print(f"    - {v}")

    return {
        "solved": solved,
        "violations": violations,
        "final_state": state,
        "num_moves": len(moves),
    }


def bfs_optimal(initial_state, goal_state, num_disks):
    """Find the optimal (shortest) solution using BFS."""
    start = state_to_tuple(initial_state)
    goal = state_to_tuple(goal_state)

    if start == goal:
        return []

    queue = deque()
    queue.append((start, []))
    visited = {start}

    while queue:
        current, path = queue.popleft()

        for from_peg in range(3):
            if not current[from_peg]:
                continue
            disk = current[from_peg][-1]

            for to_peg in range(3):
                if from_peg == to_peg:
                    continue
                if current[to_peg] and current[to_peg][-1] < disk:
                    continue

                new_state = [list(peg) for peg in current]
                new_state[from_peg] = list(current[from_peg][:-1])
                new_state[to_peg] = list(current[to_peg]) + [disk]
                new_tuple = state_to_tuple(new_state)

                if new_tuple in visited:
                    continue

                new_path = path + [[disk, from_peg, to_peg]]

                if new_tuple == goal:
                    return new_path

                visited.add(new_tuple)
                queue.append((new_tuple, new_path))

    return None


def validate_problem_file(filepath, verbose=False):
    """
    Validate a single problem JSON file.

    Returns:
        dict with summary results, or None if file is not a problem file.
    """
    with open(filepath) as f:
        problem = json.load(f)

    if "problem_info" not in problem or "response" not in problem:
        return None

    info = problem["problem_info"]
    initial_state = info["initial_state"]
    goal_state = info["goal_state"]
    num_disks = info["num_disks"]
    problem_id = info.get("problem_id", "?")
    config_type = info.get("config_type", "unknown")

    response = problem.get("response", {})
    raw = response.get("raw_content") or response.get("answer") or ""
    moves = extract_moves(raw)

    filename = os.path.basename(filepath)
    header = f"Problem {problem_id} ({filename}) — {num_disks} disks, {config_type}"

    if verbose:
        print("=" * 60)
        print(header)
        print("=" * 60)

    if moves is None:
        if verbose:
            print("  ERROR: Could not parse moves from response")
            print()
        return {
            "file": filename,
            "problem_id": problem_id,
            "num_disks": num_disks,
            "config_type": config_type,
            "parse_error": True,
            "solved": False,
            "violations": 1,
            "num_moves": 0,
            "optimal_moves": None,
            "is_optimal": False,
        }

    result = validate_moves(initial_state, goal_state, moves, num_disks, verbose=verbose)

    optimal = bfs_optimal(initial_state, goal_state, num_disks)
    optimal_len = len(optimal) if optimal is not None else None

    is_optimal = (
        result["solved"]
        and optimal_len is not None
        and result["num_moves"] == optimal_len
    )

    if verbose:
        print()
        if optimal is not None:
            print(f"  Optimal solution: {optimal_len} moves")
        else:
            print("  Optimal solution: UNSOLVABLE")
        print(f"  Model is optimal: {is_optimal}")
        if result["solved"] and optimal_len and result["num_moves"] > optimal_len:
            print(f"  Extra moves:      {result['num_moves'] - optimal_len}")
        print()

    return {
        "file": filename,
        "problem_id": problem_id,
        "num_disks": num_disks,
        "config_type": config_type,
        "parse_error": False,
        "solved": result["solved"],
        "violations": len(result["violations"]),
        "num_moves": result["num_moves"],
        "optimal_moves": optimal_len,
        "is_optimal": is_optimal,
    }


def collect_problem_files(path):
    """
    Given a path (file, run dir, or top-level results dir), return a sorted
    list of problem JSON files.
    """
    p = Path(path)

    if p.is_file() and p.suffix == ".json":
        return [str(p)]

    if p.is_dir():
        direct = sorted(p.glob("problem_*.json"))
        if direct:
            return [str(f) for f in direct]

        nested = sorted(p.glob("*/problem_*.json"))
        return [str(f) for f in nested]

    return []


def main():
    parser = argparse.ArgumentParser(
        description="Validate TOH solutions from problem JSON files."
    )
    parser.add_argument(
        "path",
        help="Path to a problem JSON file, a run directory, or the top-level results directory.",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Show step-by-step validation for each problem.",
    )
    args = parser.parse_args()

    files = collect_problem_files(args.path)
    if not files:
        print(f"No problem files found at: {args.path}")
        sys.exit(1)

    print(f"Found {len(files)} problem file(s)\n")

    results = []
    for filepath in files:
        r = validate_problem_file(filepath, verbose=args.verbose)
        if r is not None:
            results.append(r)

    if not results:
        print("No valid problem files processed.")
        sys.exit(1)

    # Save results to file
    input_path = Path(args.path)
    if input_path.is_file():
        output_dir = input_path.parent
    else:
        # For directories, find the deepest run dir
        output_dir = input_path
        if not any(input_path.glob("problem_*.json")):
            # Top-level dir — save in it directly
            pass
    results_file = output_dir / "validation_results.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {results_file}\n")

    # Summary table
    total = len(results)
    solved = sum(1 for r in results if r["solved"])
    optimal = sum(1 for r in results if r["is_optimal"])
    parse_errors = sum(1 for r in results if r["parse_error"])
    with_violations = sum(1 for r in results if r["violations"] > 0)

    print("=" * 90)
    print(f"{'File':<25} {'Disks':>5} {'Type':<13} {'Solved':>6} {'Moves':>5} {'Optimal':>7} {'Match':>6} {'Viol':>5}")
    print("-" * 90)
    for r in results:
        opt_str = str(r["optimal_moves"]) if r["optimal_moves"] is not None else "—"
        match_str = "YES" if r["is_optimal"] else ("—" if r["parse_error"] else "NO")
        solved_str = "YES" if r["solved"] else ("PARSE" if r["parse_error"] else "NO")
        print(
            f"{r['file']:<25} {r['num_disks']:>5} {r['config_type']:<13} "
            f"{solved_str:>6} {r['num_moves']:>5} {opt_str:>7} {match_str:>6} {r['violations']:>5}"
        )
    print("-" * 90)
    print(
        f"Total: {total}  |  Solved: {solved}/{total}  |  Optimal: {optimal}/{total}  |  "
        f"Parse errors: {parse_errors}  |  With violations: {with_violations}"
    )
    print("=" * 90)


if __name__ == "__main__":
    main()
