"""
Towers of Hanoi planning utilities.

This module keeps the runtime pieces used by training/eval scripts:
- State representation and move legality checks
- Optimal solver (now unified for standard and arbitrary start/goal states)
- Validators for standard and non-standard tasks
- Move parsing and constraint checking utilities
- Dataset helpers
"""

import json
import random
import re
from collections import deque
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
from torch.utils.data import Dataset

from prompts import create_standard_prompt, create_nonstandard_prompt


def _extract_moves_block(text: str) -> Optional[str]:
    last_complete_block = None

    for match in re.finditer(r'moves\s*=\s*\[', text, flags=re.IGNORECASE):
        start = text.find('[', match.start())
        if start == -1:
            continue

        depth = 0
        for idx in range(start, len(text)):
            char = text[idx]
            if char == '[':
                depth += 1
            elif char == ']':
                depth -= 1
                if depth == 0:
                    candidate = text[start:idx + 1]
                    if candidate.strip() != '[]':
                        last_complete_block = candidate
                    break

    return last_complete_block


def _parse_moves_json(moves_str: str) -> List[List[int]]:
    cleaned = re.sub(r'#[^\n]*', '', moves_str)
    cleaned = re.sub(r',\s*\]', ']', cleaned)
    return json.loads(cleaned)


# ============================================================================
# Violation Types for Constraint Loss
# ============================================================================


class ViolationType:
    """Enumeration of Towers of Hanoi rule violations."""

    DISK_NOT_ON_TOP = "disk_not_on_top"
    SOURCE_PEG_EMPTY = "source_peg_empty"
    LARGER_ON_SMALLER = "larger_on_smaller"
    INVALID_DISK_NUMBER = "invalid_disk_number"
    INVALID_PEG_NUMBER = "invalid_peg_number"
    INVALID_MOVE_FORMAT = "invalid_move_format"
    DISK_NOT_EXISTS = "disk_not_exists"


@dataclass
class MoveViolation:
    """Represents a constraint violation for a move."""

    violation_type: str
    move: Optional[List[int]]
    step_index: int
    description: str
    severity: float = 1.0


# ============================================================================
# Core State and Solver
# ============================================================================


class TowersOfHanoiState:
    """Represents a state in the Towers of Hanoi problem."""

    PEG_MAP = {"A": 0, "B": 1, "C": 2, "a": 0, "b": 1, "c": 2}

    def __init__(self, num_disks: int = 3, pegs: Optional[List[List[int]]] = None):
        self.num_disks = num_disks
        if pegs is None:
            self.pegs = [
                list(range(num_disks, 0, -1)),
                [],
                [],
            ]
        else:
            self.pegs = [list(p) for p in pegs]

    @staticmethod
    def _normalize_peg(peg: int) -> int:
        if isinstance(peg, str):
            if peg in TowersOfHanoiState.PEG_MAP:
                return TowersOfHanoiState.PEG_MAP[peg]
            if peg.isdigit() and 0 <= int(peg) <= 2:
                return int(peg)
            raise ValueError(f"Invalid peg label: {peg}")
        return int(peg)

    def copy(self):
        return TowersOfHanoiState(self.num_disks, pegs=[p.copy() for p in self.pegs])

    def is_valid_move(self, from_peg: int, to_peg: int) -> bool:
        from_peg = self._normalize_peg(from_peg)
        to_peg = self._normalize_peg(to_peg)

        if from_peg < 0 or from_peg > 2 or to_peg < 0 or to_peg > 2:
            return False
        if not self.pegs[from_peg]:
            return False

        disk = self.pegs[from_peg][-1]
        if not self.pegs[to_peg]:
            return True

        return disk < self.pegs[to_peg][-1]

    def apply_move(self, from_peg: int, to_peg: int) -> bool:
        from_peg = self._normalize_peg(from_peg)
        to_peg = self._normalize_peg(to_peg)

        if not self.is_valid_move(from_peg, to_peg):
            return False

        disk = self.pegs[from_peg].pop()
        self.pegs[to_peg].append(disk)
        return True

    def is_goal(self, goal_peg: int = 2) -> bool:
        goal_peg = self._normalize_peg(goal_peg)
        return len(self.pegs[goal_peg]) == self.num_disks

    def __str__(self):
        return f"0:{self.pegs[0]} 1:{self.pegs[1]} 2:{self.pegs[2]}"

    def __eq__(self, other):
        return self.pegs == other.pegs


class TowersOfHanoiSolver:
    """
    Unified optimal solver.

    Uses BFS over legal state transitions, so it supports both:
    - Standard tower-to-tower tasks (canonical start and goal peg)
    - Arbitrary valid start/goal configurations ("flat-to-flat")
    """

    @staticmethod
    def _state_to_tuple(state: List[List[int]]) -> Tuple[Tuple[int, ...], Tuple[int, ...], Tuple[int, ...]]:
        return tuple(tuple(peg) for peg in state)

    @staticmethod
    def _validate_state(state: List[List[int]], num_disks: int) -> None:
        if len(state) != 3:
            raise ValueError("State must have exactly 3 pegs")

        all_disks = [d for peg in state for d in peg]
        expected = list(range(1, num_disks + 1))
        if sorted(all_disks) != expected:
            raise ValueError(
                f"State must contain each disk exactly once (expected {expected}, got {sorted(all_disks)})"
            )

        for peg in state:
            for i in range(len(peg) - 1):
                if peg[i] < peg[i + 1]:
                    raise ValueError(f"Invalid peg ordering {peg}: larger disks must be below smaller disks")

    @staticmethod
    def _default_states(num_disks: int, source: int, target: int) -> Tuple[List[List[int]], List[List[int]]]:
        initial = [[], [], []]
        initial[source] = list(range(num_disks, 0, -1))

        goal = [[], [], []]
        goal[target] = list(range(num_disks, 0, -1))
        return initial, goal

    def solve(
        self,  # instance of TowersOfHanoiSolver
        num_disks: int,  # total number of disks expected in each state
        source: int = 0,  # source peg for standard mode (ignored if explicit initial/goal are passed)
        target: int = 2,  # target peg for standard mode (ignored if explicit initial/goal are passed)
        initial_state: Optional[List[List[int]]] = None,  # explicit start state (3 pegs) if provided
        goal_state: Optional[List[List[int]]] = None,  # explicit goal state (3 pegs) if provided
    ) -> Optional[List[List[int]]]:  # returns shortest move list [[disk, from, to], ...] or None
        """
        Find an optimal (shortest) path with BFS.
        """

        if initial_state is None or goal_state is None:
            # if either state is missing, build standard tower-to-tower start/goal from source/target
            initial_state, goal_state = self._default_states(num_disks, source, target)

        self._validate_state(initial_state, num_disks)
        # ensure initial state has exactly 3 pegs, proper ordering, and each disk exactly once

        self._validate_state(goal_state, num_disks)
        # same validation for goal state

        start = self._state_to_tuple(initial_state)
        # convert mutable list-of-lists into immutable tuple-of-tuples so it can be hashed

        goal = self._state_to_tuple(goal_state)
        # same conversion for goal

        if start == goal:
            # already solved: zero moves needed
            return []

        queue = deque([(start, [])])
        # BFS frontier: each item is (state, path_to_reach_state)
        # start with initial state and empty path

        visited = {start}
        # set of states already seen, to avoid cycles/reprocessing

        while queue:
            # standard BFS loop until no states left to explore
            current, path = queue.popleft()
            # pop oldest element => explores by increasing path length

            for from_peg in range(3):
                # try moving from peg 0, 1, 2
                if not current[from_peg]:
                    # if source peg empty, cannot move from it
                    continue

                disk = current[from_peg][-1]
                # top disk on source peg (only movable disk)

                for to_peg in range(3):
                    # try destination peg 0, 1, 2
                    if from_peg == to_peg:
                        # skip no-op move
                        continue

                    if current[to_peg] and current[to_peg][-1] < disk:
                        # if destination has smaller top disk, move is illegal
                        continue

                    new_state = [list(peg) for peg in current]
                    # make mutable copy of current state for simulation

                    new_state[from_peg].pop()
                    # remove top disk from source peg

                    new_state[to_peg].append(disk)
                    # place disk onto destination peg

                    new_tuple = self._state_to_tuple(new_state)
                    # convert back to immutable canonical form

                    if new_tuple in visited:
                        # skip already-seen state
                        continue

                    new_path = path + [[disk, from_peg, to_peg]]
                    # build path for this child state by appending this move

                    if new_tuple == goal:
                        # first time goal is found in BFS => guaranteed shortest path
                        return new_path

                    visited.add(new_tuple)
                    # mark as seen now (prevents duplicate enqueues)

                    queue.append((new_tuple, new_path))
                    # enqueue child for future BFS expansion

        return None
        # if queue exhausts without finding goal, no solution exists
# ============================================================================
# Validators
# ============================================================================


class TowersOfHanoiValidator:
    """Validates standard TOH responses for canonical start -> peg 2 goal."""

    def parse_moves(self, reasoning_trace: str) -> List[List[int]]:
        text_outside_think = re.sub(r'<think>.*?</think>', '', reasoning_trace, flags=re.DOTALL)
        moves_str = _extract_moves_block(text_outside_think)

        if moves_str is None:
            raise ValueError(
                "No moves found in final answer (outside <think> tags). "
                "Expected format: moves = [[disk id, from peg, to peg], ...]"
            )

        try:
            moves = _parse_moves_json(moves_str)
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse moves array: {e}")

        return moves

    def validate_trace(self, reasoning_trace: str, problem_state: Dict) -> Tuple[float, int]:
        num_disks = problem_state.get('num_disks', 3)
        goal_peg = problem_state.get('goal_peg', 2)

        state = TowersOfHanoiState(num_disks)
        if isinstance(goal_peg, str):
            goal_peg = TowersOfHanoiState._normalize_peg(goal_peg)

        goal_state = [[], [], []]
        goal_state[goal_peg] = list(range(num_disks, 0, -1))

        try:
            moves = self.parse_moves(reasoning_trace)
        except ValueError:
            return 0.0, 1

        violations = 0

        try:
            for move in moves:
                if len(move) != 3 or not all(isinstance(x, int) for x in move):
                    raise ValueError(f"Invalid move: {move}. Must be three integers.")

                disk, from_peg, to_peg = move

                if disk > num_disks or disk < 1:
                    raise ValueError(f"Invalid disk number: {disk}")
                if from_peg < 0 or from_peg > 2 or to_peg < 0 or to_peg > 2:
                    raise ValueError(f"Invalid peg index in move: {move}")

                if len(state.pegs[from_peg]) == 0 or state.pegs[from_peg][-1] != disk:
                    raise ValueError(f"From peg {from_peg} does not contain disk {disk} on top")

                if len(state.pegs[to_peg]) > 0 and state.pegs[to_peg][-1] < disk:
                    raise ValueError(
                        f"Cannot place disk {disk} on top of disk {state.pegs[to_peg][-1]}"
                    )

                state.pegs[to_peg].append(state.pegs[from_peg].pop())

        except ValueError:
            violations += 1

        solved = state.pegs == goal_state

        if solved:
            optimal_moves = 2 ** num_disks - 1
            if len(moves) == optimal_moves:
                reward = 1.5
            elif len(moves) <= optimal_moves * 1.2:
                reward = 1.2
            else:
                reward = 1.0
        else:
            reward = 0.0

        return reward, violations

    def batch_validate(self, traces: List[str], problem_states: List[Dict]) -> Tuple[torch.Tensor, torch.Tensor]:
        rewards = []
        violations = []

        for trace, state in zip(traces, problem_states):
            r, v = self.validate_trace(trace, state)
            rewards.append(r)
            violations.append(v)

        return torch.tensor(rewards), torch.tensor(violations)


class NonStandardValidator:
    """Validates solutions for arbitrary initial/goal TOH configurations."""

    def __init__(self):
        self.solver = TowersOfHanoiSolver()

    def parse_moves(self, response: str) -> Optional[List[List[int]]]:
        moves_str = _extract_moves_block(response)
        if moves_str is None:
            return None

        try:
            moves = _parse_moves_json(moves_str)
            return moves
        except json.JSONDecodeError:
            return None

    def validate(
        self,
        response: str,
        initial_state: List[List[int]],
        goal_state: List[List[int]],
        num_disks: int,
    ) -> Dict:
        moves = self.parse_moves(response)

        if moves is None:
            return {
                'success': False,
                'error': 'Failed to parse moves',
                'violations': 1,
                'num_moves': 0,
                'solved': False,
            }

        state = [peg.copy() for peg in initial_state]
        violations = 0

        for move in moves:
            if len(move) != 3:
                violations += 1
                continue

            disk, from_peg, to_peg = move

            if disk < 1 or disk > num_disks:
                violations += 1
                continue
            if from_peg < 0 or from_peg > 2 or to_peg < 0 or to_peg > 2:
                violations += 1
                continue
            if not state[from_peg] or state[from_peg][-1] != disk:
                violations += 1
                continue
            if state[to_peg] and state[to_peg][-1] < disk:
                violations += 1
                continue

            state[from_peg].pop()
            state[to_peg].append(disk)

        solved = state == goal_state

        optimal_path = None
        optimal_moves = None
        is_optimal = False
        try:
            optimal_path = self.solver.solve(
                num_disks=num_disks,
                initial_state=initial_state,
                goal_state=goal_state,
            )
            optimal_moves = len(optimal_path) if optimal_path is not None else None
            is_optimal = (
                solved
                and violations == 0
                and optimal_moves is not None
                and len(moves) == optimal_moves
            )
        except ValueError:
            optimal_path = None
            optimal_moves = None
            is_optimal = False

        return {
            'success': True,
            'violations': violations,
            'num_moves': len(moves),
            'solved': solved,
            'final_state': state,
            'optimal_moves': optimal_moves,
            'is_optimal': is_optimal,
            'extra_moves': (len(moves) - optimal_moves) if (solved and optimal_moves is not None) else None,
        }


# ============================================================================
# Move Parser and Constraint Checker
# ============================================================================


class MoveParser:
    """Parses moves from responses."""

    def __init__(self):
        self.move_patterns = [
            re.compile(
                r'[Mm]ove\s+[Dd]isk\s+(\d+)\s+from\s+[Pp]eg\s+(\d+)\s+to\s+[Pp]eg\s+(\d+)',
                re.IGNORECASE,
            ),
            re.compile(r'\[\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\]'),
            re.compile(r'[Dd]isk\s+(\d+)[:\s]+(\d+)\s*(?:->|→|to)\s*(\d+)'),
        ]

        self.final_moves_pattern = re.compile(
            r'moves\s*=\s*(\[(?:\s*\[[^\]]+\]\s*,?\s*)+\])',
            re.DOTALL,
        )

    def parse_final_moves(self, text: str) -> Optional[List[List[int]]]:
        text_outside_think = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
        matches = self.final_moves_pattern.findall(text_outside_think)
        if not matches:
            return None

        moves_str = matches[-1].strip()
        try:
            moves = json.loads(moves_str)
            return moves
        except json.JSONDecodeError:
            return None

    def parse_intermediate_moves(self, text: str) -> List[Tuple[int, List[int]]]:
        moves = []
        for pattern in self.move_patterns:
            for match in pattern.finditer(text):
                try:
                    disk = int(match.group(1))
                    from_peg = int(match.group(2))
                    to_peg = int(match.group(3))
                    moves.append((match.start(), [disk, from_peg, to_peg]))
                except (ValueError, IndexError):
                    continue

        moves.sort(key=lambda x: x[0])
        return moves

    def extract_reasoning_moves(self, text: str) -> List[List[int]]:
        think_match = re.search(r'<think>(.*?)</think>', text, re.DOTALL)
        reasoning = think_match.group(1) if think_match else text
        intermediate = self.parse_intermediate_moves(reasoning)
        return [move for _, move in intermediate]


class ConstraintChecker:
    """Checks Towers of Hanoi constraints and computes violation penalties."""

    def __init__(self, num_disks: int):
        self.num_disks = num_disks
        self.initial_state = TowersOfHanoiState(num_disks)

    def check_move(
        self,
        move: List[int],
        state: TowersOfHanoiState,
        step_index: int,
    ) -> List[MoveViolation]:
        violations = []

        if len(move) != 3:
            violations.append(
                MoveViolation(
                    violation_type=ViolationType.INVALID_MOVE_FORMAT,
                    move=move,
                    step_index=step_index,
                    description=f"Move must have 3 elements, got {len(move)}",
                    severity=1.0,
                )
            )
            return violations

        disk, from_peg, to_peg = move

        if not isinstance(disk, int) or disk < 1 or disk > self.num_disks:
            violations.append(
                MoveViolation(
                    violation_type=ViolationType.INVALID_DISK_NUMBER,
                    move=move,
                    step_index=step_index,
                    description=f"Invalid disk number {disk}, must be 1-{self.num_disks}",
                    severity=1.0,
                )
            )
            return violations

        if not isinstance(from_peg, int) or from_peg < 0 or from_peg > 2:
            violations.append(
                MoveViolation(
                    violation_type=ViolationType.INVALID_PEG_NUMBER,
                    move=move,
                    step_index=step_index,
                    description=f"Invalid source peg {from_peg}, must be 0-2",
                    severity=1.0,
                )
            )
            return violations

        if not isinstance(to_peg, int) or to_peg < 0 or to_peg > 2:
            violations.append(
                MoveViolation(
                    violation_type=ViolationType.INVALID_PEG_NUMBER,
                    move=move,
                    step_index=step_index,
                    description=f"Invalid destination peg {to_peg}, must be 0-2",
                    severity=1.0,
                )
            )
            return violations

        if len(state.pegs[from_peg]) == 0:
            violations.append(
                MoveViolation(
                    violation_type=ViolationType.SOURCE_PEG_EMPTY,
                    move=move,
                    step_index=step_index,
                    description=f"Source peg {from_peg} is empty",
                    severity=1.0,
                )
            )
            return violations

        top_disk = state.pegs[from_peg][-1]
        if top_disk != disk:
            violations.append(
                MoveViolation(
                    violation_type=ViolationType.DISK_NOT_ON_TOP,
                    move=move,
                    step_index=step_index,
                    description=f"Disk {disk} is not on top of peg {from_peg}, top disk is {top_disk}",
                    severity=1.0,
                )
            )
            return violations

        if len(state.pegs[to_peg]) > 0:
            top_dest = state.pegs[to_peg][-1]
            if disk > top_dest:
                violations.append(
                    MoveViolation(
                        violation_type=ViolationType.LARGER_ON_SMALLER,
                        move=move,
                        step_index=step_index,
                        description=f"Cannot place disk {disk} on smaller disk {top_dest}",
                        severity=1.0,
                    )
                )

        return violations

    def check_move_sequence(self, moves: List[List[int]]) -> Tuple[List[MoveViolation], TowersOfHanoiState]:
        state = self.initial_state.copy()
        all_violations = []

        for i, move in enumerate(moves):
            violations = self.check_move(move, state, i)
            all_violations.extend(violations)

            if not violations and len(move) == 3:
                disk, from_peg, to_peg = move
                if len(state.pegs[from_peg]) > 0 and state.pegs[from_peg][-1] == disk:
                    state.pegs[from_peg].pop()
                    state.pegs[to_peg].append(disk)

        return all_violations, state

    def compute_violation_score(self, violations: List[MoveViolation]) -> float:
        if not violations:
            return 0.0

        total_severity = sum(v.severity for v in violations)
        expected_moves = 2 ** self.num_disks - 1
        normalized = total_severity / max(expected_moves, 1)
        return min(normalized, 1.0)


# ============================================================================
# Datasets
# ============================================================================


class TowersOfHanoiDataset:
    """Generates standard Towers of Hanoi problems and optimal solutions."""

    def __init__(self, min_disks: int = 2, max_disks: int = 5):
        self.min_disks = min_disks
        self.max_disks = max_disks
        self.solver = TowersOfHanoiSolver()

    def generate_problem(self, num_disks: Optional[int] = None) -> Dict:
        if num_disks is None:
            num_disks = random.randint(self.min_disks, self.max_disks)

        goal_peg = 2

        initial_state = TowersOfHanoiState(num_disks)
        system_prompt, user_prompt = create_standard_prompt(num_disks, goal_peg=goal_peg)

        optimal_moves = self.solver.solve(num_disks, source=0, target=goal_peg)

        return {
            'num_disks': num_disks,
            'initial_state': initial_state,
            'goal_peg': goal_peg,
            'system_prompt': system_prompt,
            'user_prompt': user_prompt,
            'problem_text': user_prompt,
            'optimal_moves': optimal_moves,
        }

    def generate_batch(self, batch_size: int) -> List[Dict]:
        return [self.generate_problem() for _ in range(batch_size)]


class TOHDataset(Dataset):
    """Dataset for Towers of Hanoi GRPO training with equal disk proportions."""

    def __init__(
        self,
        num_problems: int,
        min_disks: int = 3,
        max_disks: int = 5,
    ):
        self.num_problems = num_problems
        self.min_disks = min_disks
        self.max_disks = max_disks
        self.problems = self._generate_problems()

    def _generate_problems(self) -> List[Dict]:
        problems = []
        dataset = TowersOfHanoiDataset(self.min_disks, self.max_disks)

        disk_counts = list(range(self.min_disks, self.max_disks + 1))
        num_per_disk = self.num_problems // len(disk_counts)
        remainder = self.num_problems % len(disk_counts)

        for i, num_disks in enumerate(disk_counts):
            count = num_per_disk + (1 if i < remainder else 0)

            for _ in range(count):
                problem = dataset.generate_problem(num_disks=num_disks)

                problems.append({
                    'num_disks': num_disks,
                    'goal_peg': 2,
                    'system_prompt': problem['system_prompt'],
                    'user_prompt': problem['user_prompt'],
                    'optimal_moves': 2 ** num_disks - 1,
                })

        random.shuffle(problems)
        return problems

    def __len__(self) -> int:
        return self.num_problems

    def __getitem__(self, idx: int) -> Dict:
        return self.problems[idx]


class NonStandardTOHDataset(Dataset):
    """
    Dataset for non-standard (flat-to-flat) Towers of Hanoi problems.

    Initial and goal states are arbitrary valid configurations with all n disks
    distributed across pegs while respecting ordering constraints.
    """

    def __init__(
        self,
        num_problems: int,
        min_disks: int = 3,
        max_disks: int = 5,
        seed: int = 42,
    ):
        self.num_problems = num_problems
        self.min_disks = min_disks
        self.max_disks = max_disks
        self.seed = seed
        self.solver = TowersOfHanoiSolver()
        self.problems = self._generate_problems()

    def _generate_problems(self) -> List[Dict]:
        problems = []

        disk_counts = list(range(self.min_disks, self.max_disks + 1))
        num_per_disk = self.num_problems // len(disk_counts)
        remainder = self.num_problems % len(disk_counts)

        problem_id = 0
        for i, num_disks in enumerate(disk_counts):
            count = num_per_disk + (1 if i < remainder else 0)

            for local_id in range(count):
                system_prompt, user_prompt, problem_info = create_nonstandard_prompt(
                    num_disks=num_disks,
                    problem_id=local_id,
                    seed=self.seed,
                )

                optimal_moves = self.solver.solve(
                    num_disks=num_disks,
                    initial_state=problem_info['initial_state'],
                    goal_state=problem_info['goal_state'],
                )

                problems.append({
                    'problem_id': problem_id,
                    'num_disks': num_disks,
                    'system_prompt': system_prompt,
                    'user_prompt': user_prompt,
                    'problem_info': problem_info,
                    'initial_state': problem_info['initial_state'],
                    'goal_state': problem_info['goal_state'],
                    'config_type': 'nonstandard',
                    'optimal_moves': optimal_moves if optimal_moves is not None else [],
                })
                problem_id += 1

        random.shuffle(problems)
        return problems

    def __len__(self) -> int:
        return self.num_problems

    def __getitem__(self, idx: int) -> Dict:
        return self.problems[idx]


class BaselineProblemGenerator:
    """Generates baseline-evaluation problem sets for standard or non-standard TOH."""

    @staticmethod
    def generate_problems(
        num_problems: int,
        min_disks: int = 3,
        max_disks: int = 5,
        seed: int = 42,
        mode: str = "nonstandard",
    ) -> List[Dict]:
        mode = mode.strip().lower()
        if mode not in {"standard", "nonstandard"}:
            raise ValueError(f"Invalid mode '{mode}'; expected 'standard' or 'nonstandard'")

        if mode == "nonstandard":
            dataset = NonStandardTOHDataset(
                num_problems=num_problems,
                min_disks=min_disks,
                max_disks=max_disks,
                seed=seed,
            )
            return dataset.problems

        dataset = TowersOfHanoiDataset(min_disks=min_disks, max_disks=max_disks)
        disk_counts = list(range(min_disks, max_disks + 1))
        num_per_disk = num_problems // len(disk_counts)
        remainder = num_problems % len(disk_counts)

        problems = []
        problem_id = 0
        for i, num_disks in enumerate(disk_counts):
            count = num_per_disk + (1 if i < remainder else 0)

            for local_id in range(count):
                problem = dataset.generate_problem(num_disks=num_disks)
                initial_state = [list(range(num_disks, 0, -1)), [], []]
                goal_peg = problem["goal_peg"]
                goal_state = [[], [], []]
                goal_state[goal_peg] = list(range(num_disks, 0, -1))

                problem_info = {
                    "num_disks": num_disks,
                    "initial_state": initial_state,
                    "goal_state": goal_state,
                    "config_type": "standard",
                    "goal_peg": goal_peg,
                    "problem_id": local_id,
                    "problem_seed": seed,
                }

                problems.append({
                    "problem_id": problem_id,
                    "num_disks": num_disks,
                    "system_prompt": problem["system_prompt"],
                    "user_prompt": problem["user_prompt"],
                    "problem_info": problem_info,
                    "initial_state": initial_state,
                    "goal_state": goal_state,
                    "config_type": "standard",
                    "goal_peg": goal_peg,
                    "optimal_moves": problem["optimal_moves"] if problem["optimal_moves"] is not None else [],
                })
                problem_id += 1

        random.shuffle(problems)
        return problems


if __name__ == "__main__":
    solver = TowersOfHanoiSolver()

    print("Standard 3-disk (tower-to-tower):")
    std = solver.solve(num_disks=3)
    print(std)
    print(f"Moves: {len(std)}")

    print("\nFlat-to-flat example:")
    initial = [[4, 1], [3], [2]]
    goal = [[4], [3, 2, 1], []]
    flat = solver.solve(num_disks=4, initial_state=initial, goal_state=goal)
    print(flat)
    print(f"Moves: {len(flat) if flat is not None else 'unsolved'}")
