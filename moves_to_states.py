import argparse
import ast
from typing import List


def label_from_pegs(pegs: List[List[int]], num_disks: int) -> str:
    disk_to_peg = {}
    for peg_idx, peg in enumerate(pegs):
        for disk in peg:
            disk_to_peg[disk] = peg_idx

    # User-requested convention: largest -> smallest (disk n ... disk 1)
    return "".join(str(disk_to_peg[disk] + 1) for disk in range(num_disks, 0, -1))


def apply_moves_and_collect_states(moves: List[List[int]], num_disks: int) -> List[str]:
    pegs = [list(range(num_disks, 0, -1)), [], []]
    states = [label_from_pegs(pegs, num_disks)]

    for step, move in enumerate(moves, start=1):
        if len(move) != 3:
            raise ValueError(f"Step {step}: invalid move format {move}")

        disk, from_peg, to_peg = move
        if from_peg not in (0, 1, 2) or to_peg not in (0, 1, 2):
            raise ValueError(f"Step {step}: peg must be 0..2, got {move}")
        if disk < 1 or disk > num_disks:
            raise ValueError(f"Step {step}: invalid disk id {disk}")

        if not pegs[from_peg] or pegs[from_peg][-1] != disk:
            raise ValueError(
                f"Step {step}: disk {disk} is not on top of peg {from_peg}. "
                f"Current pegs={pegs}"
            )

        if pegs[to_peg] and pegs[to_peg][-1] < disk:
            raise ValueError(
                f"Step {step}: cannot place disk {disk} on smaller disk {pegs[to_peg][-1]}"
            )

        pegs[to_peg].append(pegs[from_peg].pop())
        states.append(label_from_pegs(pegs, num_disks))

    return states


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert [disk, from_peg, to_peg] moves to successive state labels."
    )
    parser.add_argument(
        "--moves",
        required=True,
        help="Python-style list of moves, e.g. '[[1,0,2],[2,0,1]]'",
    )
    parser.add_argument(
        "--num-disks",
        type=int,
        default=None,
        help="Number of disks (default: inferred from max disk id in moves)",
    )
    args = parser.parse_args()

    moves = ast.literal_eval(args.moves)
    if not isinstance(moves, list) or not moves:
        raise ValueError("--moves must be a non-empty list")

    inferred = max(m[0] for m in moves)
    num_disks = args.num_disks if args.num_disks is not None else inferred

    states = apply_moves_and_collect_states(moves, num_disks)

    print(" -> ".join(states))
    print("\nIndexed states:")
    for idx, state in enumerate(states):
        print(f"{idx:2d}: {state}")


if __name__ == "__main__":
    main()
