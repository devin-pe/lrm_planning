"""Data generation and tokenization for flat-to-flat Towers of Hanoi."""

from __future__ import annotations

import itertools
import random
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import torch
from torch.utils.data import Dataset

from planning import TowersOfHanoiSolver

StateTuple = Tuple[int, ...]
MovePair = Tuple[int, int]
PairKey = Tuple[StateTuple, StateTuple]

SPECIAL_TOKENS = ["BOS", "SEP", "EOS", "PAD"]
POSITION_TOKENS = ["P0", "P1", "P2"]
MOVE_TOKENS = ["M01", "M02", "M10", "M12", "M20", "M21"]
ALL_TOKENS = POSITION_TOKENS + MOVE_TOKENS + SPECIAL_TOKENS


class Vocabulary:
    def __init__(self) -> None:
        self.itos: List[str] = list(ALL_TOKENS)
        self.stoi: Dict[str, int] = {tok: i for i, tok in enumerate(self.itos)}

        self.pad_id = self.stoi["PAD"]
        self.bos_id = self.stoi["BOS"]
        self.sep_id = self.stoi["SEP"]
        self.eos_id = self.stoi["EOS"]
        self.move_token_ids = {self.stoi[t] for t in MOVE_TOKENS}
        self.loss_token_ids = set(self.move_token_ids) | {self.eos_id}

    def __len__(self) -> int:
        return len(self.itos)


def tuple_to_pegs(state_tuple: StateTuple, n_disks: int) -> List[List[int]]:
    """Convert tuple state into planning.py peg format (bottom->top disk ids).

    Note: disk id in planning.py is 1..n where larger id means larger disk.
    Since tuple index 0 is smallest disk, disk_id = disk_idx + 1.
    """
    pegs = [[] for _ in range(3)]
    for disk_idx in range(n_disks - 1, -1, -1):
        disk_id = disk_idx + 1
        pegs[state_tuple[disk_idx]].append(disk_id)
    return [list(p) for p in pegs]


def enumerate_all_states(n_disks: int) -> List[StateTuple]:
    return [tuple(s) for s in itertools.product((0, 1, 2), repeat=n_disks)]


def move_triplet_to_pair(move: Sequence[int]) -> MovePair:
    if len(move) != 3:
        raise ValueError(f"Unexpected move format: {move}")
    return int(move[1]), int(move[2])


def move_pair_to_token(move: MovePair) -> str:
    return f"M{move[0]}{move[1]}"


def state_to_tokens(state: StateTuple) -> List[str]:
    return [f"P{peg}" for peg in state]


def build_solution_cache(n_disks: int) -> Dict[PairKey, Tuple[MovePair, ...]]:
    """Solve all start/goal pairs with planning.TowersOfHanoiSolver (no BFS reimplementation)."""
    solver = TowersOfHanoiSolver()
    states = enumerate_all_states(n_disks)
    cache: Dict[PairKey, Tuple[MovePair, ...]] = {}

    for start in states:
        start_pegs = tuple_to_pegs(start, n_disks)
        for goal in states:
            goal_pegs = tuple_to_pegs(goal, n_disks)
            solution = solver.solve(
                num_disks=n_disks,
                initial_state=start_pegs,
                goal_state=goal_pegs,
            )
            if solution is None:
                raise RuntimeError(f"No solution found for start={start}, goal={goal}")
            cache[(start, goal)] = tuple(move_triplet_to_pair(m) for m in solution)

    return cache


def build_canonical_tower_example(n_disks: int) -> Tuple[StateTuple, StateTuple, Tuple[MovePair, ...]]:
    """Build canonical all-on-peg-0 -> all-on-peg-2 example for a given n."""
    solver = TowersOfHanoiSolver()

    start: StateTuple = tuple(0 for _ in range(n_disks))
    goal: StateTuple = tuple(2 for _ in range(n_disks))

    solution = solver.solve(
        num_disks=n_disks,
        initial_state=tuple_to_pegs(start, n_disks),
        goal_state=tuple_to_pegs(goal, n_disks),
    )
    if solution is None:
        raise RuntimeError(f"No canonical solution found for n_disks={n_disks}")

    moves = tuple(move_triplet_to_pair(m) for m in solution)
    return start, goal, moves


def _top_disk_index_on_peg(state: StateTuple, peg: int) -> int | None:
    for disk_idx, p in enumerate(state):
        if p == peg:
            return disk_idx
    return None


def _apply_move_to_state(state: StateTuple, move: MovePair) -> StateTuple:
    src, dst = move
    moving = _top_disk_index_on_peg(state, src)
    if moving is None:
        raise RuntimeError(f"Illegal move from empty peg: move={move}, state={state}")

    top_dst = _top_disk_index_on_peg(state, dst)
    if top_dst is not None and moving > top_dst:
        raise RuntimeError(f"Illegal move placing larger on smaller: move={move}, state={state}")

    out = list(state)
    out[moving] = dst
    return tuple(out)


def build_recursive_tower_examples(n_disks: int) -> List[Tuple[StateTuple, StateTuple, Tuple[MovePair, ...]]]:
    """Build recursive subproblems from states along one optimal canonical trajectory.

    Each example uses an intermediate trajectory state as start and keeps the same
    final goal (all disks on peg 2). Targets are the remaining optimal suffix moves.
    """
    start, goal, moves = build_canonical_tower_example(n_disks)

    states: List[StateTuple] = [start]
    cur = start
    for mv in moves:
        cur = _apply_move_to_state(cur, mv)
        states.append(cur)

    if states[-1] != goal:
        raise RuntimeError(f"Canonical trajectory did not end at goal for n_disks={n_disks}")

    examples: List[Tuple[StateTuple, StateTuple, Tuple[MovePair, ...]]] = []
    for idx in range(len(moves)):
        examples.append((states[idx], goal, moves[idx:]))
    return examples


def split_pairs_with_state_coverage(
    pair_items: List[Tuple[PairKey, Tuple[MovePair, ...]]],
    seed: int,
    train_ratio: float = 0.8,
) -> Tuple[List[Tuple[PairKey, Tuple[MovePair, ...]]], List[Tuple[PairKey, Tuple[MovePair, ...]]]]:
    """Split pairs while guaranteeing all states remain represented in train.

    Guarantee: every state appears at least once as a start and at least once as a goal
    in the training split.
    """
    rng = random.Random(seed)
    shuffled = pair_items.copy()
    rng.shuffle(shuffled)

    target_train_size = int(len(shuffled) * train_ratio)
    target_val_size = len(shuffled) - target_train_size

    # Start with everything in train, then move safe pairs to val.
    start_counts: Dict[StateTuple, int] = {}
    goal_counts: Dict[StateTuple, int] = {}
    for (start, goal), _ in shuffled:
        start_counts[start] = start_counts.get(start, 0) + 1
        goal_counts[goal] = goal_counts.get(goal, 0) + 1

    train: List[Tuple[PairKey, Tuple[MovePair, ...]]] = []
    val: List[Tuple[PairKey, Tuple[MovePair, ...]]] = []

    for item in shuffled:
        if len(val) >= target_val_size:
            train.append(item)
            continue

        (start, goal), _moves = item

        # A pair is safe to move to val only if train still keeps at least one
        # outgoing example for `start` and one incoming example for `goal`.
        if start_counts[start] > 1 and goal_counts[goal] > 1:
            val.append(item)
            start_counts[start] -= 1
            goal_counts[goal] -= 1
        else:
            train.append(item)

    # In rare cases, top up val from train if possible while preserving coverage.
    if len(val) < target_val_size:
        remaining_train: List[Tuple[PairKey, Tuple[MovePair, ...]]] = []
        for item in train:
            if len(val) >= target_val_size:
                remaining_train.append(item)
                continue

            (start, goal), _moves = item
            if start_counts[start] > 1 and goal_counts[goal] > 1:
                val.append(item)
                start_counts[start] -= 1
                goal_counts[goal] -= 1
            else:
                remaining_train.append(item)
        train = remaining_train

    return train, val


@dataclass
class EncodedExample:
    input_ids: torch.Tensor
    target_ids: torch.Tensor
    loss_mask: torch.Tensor
    prefix_ids: List[int]
    move_target_ids: List[int]


def encode_example(
    start: StateTuple,
    goal: StateTuple,
    moves: Sequence[MovePair],
    vocab: Vocabulary,
    max_seq_len: int,
) -> EncodedExample:
    seq_tokens = (
        ["BOS"]
        + state_to_tokens(start)
        + ["SEP"]
        + state_to_tokens(goal)
        + ["SEP"]
        + [move_pair_to_token(m) for m in moves]
        + ["EOS"]
    )

    if len(seq_tokens) > max_seq_len:
        raise ValueError(
            f"Sequence length {len(seq_tokens)} exceeds max_seq_len={max_seq_len}. "
            f"start={start}, goal={goal}, moves={len(moves)}"
        )

    ids = [vocab.stoi[tok] for tok in seq_tokens]
    prefix_len = 1 + len(start) + 1 + len(goal) + 1
    prefix_ids = ids[:prefix_len]
    move_target_ids = ids[prefix_len:]

    padded_ids = ids + [vocab.pad_id] * (max_seq_len - len(ids))

    input_ids = torch.tensor(padded_ids, dtype=torch.long)
    target_ids = torch.tensor(padded_ids[1:] + [vocab.pad_id], dtype=torch.long)
    loss_mask = torch.tensor([tid in vocab.loss_token_ids for tid in target_ids.tolist()], dtype=torch.bool)

    return EncodedExample(
        input_ids=input_ids,
        target_ids=target_ids,
        loss_mask=loss_mask,
        prefix_ids=prefix_ids,
        move_target_ids=move_target_ids,
    )


class ToHFlatDataset(Dataset):
    def __init__(
        self,
        examples: Sequence[Tuple[StateTuple, StateTuple, Tuple[MovePair, ...]]],
        vocab: Vocabulary,
        max_seq_len: int,
    ) -> None:
        self.encoded: List[EncodedExample] = [
            encode_example(start, goal, moves, vocab=vocab, max_seq_len=max_seq_len)
            for (start, goal, moves) in examples
        ]

    def __len__(self) -> int:
        return len(self.encoded)

    def __getitem__(self, idx: int):
        ex = self.encoded[idx]
        return ex.input_ids, ex.target_ids, ex.loss_mask


@dataclass
class DatasetBundle:
    train_dataset: ToHFlatDataset
    val_dataset: ToHFlatDataset | None
    test_dataset: ToHFlatDataset | None
    vocab: Vocabulary
    max_seq_len: int


def build_datasets(n_disks: int, seed: int = 42) -> DatasetBundle:
    if n_disks == 3:
        max_seq_len = 40
    elif n_disks == 4:
        max_seq_len = 96
    else:
        raise ValueError("Only n_disks=3 or n_disks=4 are supported")

    cache = build_solution_cache(n_disks)
    pair_items = list(cache.items())

    if n_disks == 3:
        train_examples = [(k[0], k[1], v) for k, v in pair_items]
        val_examples = None
    else:
        train_pairs, val_pairs = split_pairs_with_state_coverage(pair_items, seed=seed, train_ratio=0.8)
        train_examples = [(k[0], k[1], v) for k, v in train_pairs]
        val_examples = [(k[0], k[1], v) for k, v in val_pairs]

    vocab = Vocabulary()
    train_dataset = ToHFlatDataset(train_examples, vocab=vocab, max_seq_len=max_seq_len)
    val_dataset = None if val_examples is None else ToHFlatDataset(val_examples, vocab=vocab, max_seq_len=max_seq_len)

    return DatasetBundle(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        test_dataset=None,
        vocab=vocab,
        max_seq_len=max_seq_len,
    )


def build_multi_n_datasets(
    train_n_disks: Sequence[int],
    test_n_disks: Sequence[int],
    max_seq_len: int,
    seed: int = 42,
    val_ratio: float = 0.2,
) -> DatasetBundle:
    if not train_n_disks:
        raise ValueError("train_n_disks must not be empty")

    rng = random.Random(seed)

    train_examples: List[Tuple[StateTuple, StateTuple, Tuple[MovePair, ...]]] = []
    val_examples: List[Tuple[StateTuple, StateTuple, Tuple[MovePair, ...]]] = []
    test_examples: List[Tuple[StateTuple, StateTuple, Tuple[MovePair, ...]]] = []

    for n in train_n_disks:
        cache = build_solution_cache(n)
        pair_items = list(cache.items())

        if val_ratio <= 0.0:
            train_pairs = pair_items
            split_val_pairs: List[Tuple[PairKey, Tuple[MovePair, ...]]] = []
        elif val_ratio >= 1.0:
            raise ValueError("val_ratio must satisfy 0 <= val_ratio < 1")
        else:
            train_pairs, split_val_pairs = split_pairs_with_state_coverage(
                pair_items,
                seed=rng.randint(0, 10**9),
                train_ratio=1.0 - val_ratio,
            )

        train_examples.extend((k[0], k[1], v) for k, v in train_pairs)
        val_examples.extend((k[0], k[1], v) for k, v in split_val_pairs)

    for n in test_n_disks:
        cache = build_solution_cache(n)
        pair_items = list(cache.items())
        test_examples.extend((k[0], k[1], v) for k, v in pair_items)

    vocab = Vocabulary()
    train_dataset = ToHFlatDataset(train_examples, vocab=vocab, max_seq_len=max_seq_len)
    val_dataset = None if len(val_examples) == 0 else ToHFlatDataset(val_examples, vocab=vocab, max_seq_len=max_seq_len)
    test_dataset = (
        None if len(test_examples) == 0 else ToHFlatDataset(test_examples, vocab=vocab, max_seq_len=max_seq_len)
    )

    return DatasetBundle(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        test_dataset=test_dataset,
        vocab=vocab,
        max_seq_len=max_seq_len,
    )


def build_tower_to_tower_multi_n_datasets(
    train_n_disks: Sequence[int],
    test_n_disks: Sequence[int],
    max_seq_len: int,
    seed: int = 42,
    val_ratio: float = 0.2,
    train_repeats: int = 1024,
    test_repeats: int = 1,
    data_strategy: str = "canonical-repeat",
) -> DatasetBundle:
    if not train_n_disks:
        raise ValueError("train_n_disks must not be empty")
    if train_repeats < 1:
        raise ValueError("train_repeats must be >= 1")
    if test_repeats < 1:
        raise ValueError("test_repeats must be >= 1")

    rng = random.Random(seed)

    train_examples: List[Tuple[StateTuple, StateTuple, Tuple[MovePair, ...]]] = []
    val_examples: List[Tuple[StateTuple, StateTuple, Tuple[MovePair, ...]]] = []
    test_examples: List[Tuple[StateTuple, StateTuple, Tuple[MovePair, ...]]] = []

    for n in train_n_disks:
        if data_strategy == "recursive-states":
            repeated = build_recursive_tower_examples(n)
        else:
            ex = build_canonical_tower_example(n)
            repeated = [ex for _ in range(train_repeats)]
        rng.shuffle(repeated)

        if val_ratio <= 0.0:
            train_split = repeated
            val_split: List[Tuple[StateTuple, StateTuple, Tuple[MovePair, ...]]] = []
        elif val_ratio >= 1.0:
            raise ValueError("val_ratio must satisfy 0 <= val_ratio < 1")
        else:
            n_val = int(round(len(repeated) * val_ratio))
            n_val = max(1, min(n_val, len(repeated) - 1))
            val_split = repeated[:n_val]
            train_split = repeated[n_val:]

        train_examples.extend(train_split)
        val_examples.extend(val_split)

    for n in test_n_disks:
        if data_strategy == "recursive-states":
            test_examples.extend(build_recursive_tower_examples(n))
        else:
            ex = build_canonical_tower_example(n)
            test_examples.extend(ex for _ in range(test_repeats))

    vocab = Vocabulary()
    train_dataset = ToHFlatDataset(train_examples, vocab=vocab, max_seq_len=max_seq_len)
    val_dataset = None if len(val_examples) == 0 else ToHFlatDataset(val_examples, vocab=vocab, max_seq_len=max_seq_len)
    test_dataset = (
        None if len(test_examples) == 0 else ToHFlatDataset(test_examples, vocab=vocab, max_seq_len=max_seq_len)
    )

    return DatasetBundle(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        test_dataset=test_dataset,
        vocab=vocab,
        max_seq_len=max_seq_len,
    )
