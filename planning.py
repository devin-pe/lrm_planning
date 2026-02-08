"""
Towers of Hanoi Planning Validator and Expert Solver

This module implements:
1. TowersOfHanoiValidator: Validates reasoning traces and computes rewards
2. TowersOfHanoiSolver: Generates optimal solutions for supervision
3. State tracking and constraint checking for the Towers of Hanoi domain
4. MoveParser: Parses moves from reasoning traces
5. ConstraintChecker: Checks Towers of Hanoi constraints
6. TOHDataset: Training dataset for GRPO
"""

import re
import json
import random
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import torch
from torch.utils.data import Dataset
from copy import deepcopy

from prompts import create_standard_prompt


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
    severity: float = 1.0  # Weight for the violation


class TowersOfHanoiState:
    """Represents a state in the Towers of Hanoi problem."""
    
    def __init__(self, num_disks: int = 3):
        """
        Initialize with all disks on peg 0.
        Disks are numbered 1 (smallest) to num_disks (largest).
        """
        self.num_disks = num_disks
        self.pegs = [
            list(range(num_disks, 0, -1)),  # [3, 2, 1] for 3 disks (top is rightmost)
            [],
            []
        ]
    
    def copy(self):
        """Create a deep copy of the state."""
        new_state = TowersOfHanoiState(self.num_disks)
        new_state.pegs = [peg.copy() for peg in self.pegs]
        return new_state
    
    def is_valid_move(self, from_peg: int, to_peg: int) -> bool:
        """
        Check if a move is valid according to Towers of Hanoi rules.
        
        Rules:
        1. Can only move the top disk from a peg
        2. Cannot place a larger disk on a smaller disk
        """
        if from_peg < 0 or from_peg > 2 or to_peg < 0 or to_peg > 2:
            return False
        
        if not self.pegs[from_peg]:  # Source peg is empty
            return False
        
        disk = self.pegs[from_peg][-1]  # Top disk
        
        if not self.pegs[to_peg]:  # Destination peg is empty
            return True
        
        top_dest = self.pegs[to_peg][-1]
        return disk < top_dest  # Smaller disk on larger disk
    
    def apply_move(self, from_peg: int, to_peg: int) -> bool:
        """
        Apply a move to the state. Returns True if successful, False otherwise.
        """
        if not self.is_valid_move(from_peg, to_peg):
            return False
        
        disk = self.pegs[from_peg].pop()
        self.pegs[to_peg].append(disk)
        return True
    
    def is_goal(self, goal_peg: int = 2) -> bool:
        """Check if all disks are on the goal peg."""
        return len(self.pegs[goal_peg]) == self.num_disks
    
    def __str__(self):
        return f"0:{self.pegs[0]} 1:{self.pegs[1]} 2:{self.pegs[2]}"
    
    def __eq__(self, other):
        return self.pegs == other.pegs


class TowersOfHanoiSolver:
    """Generates optimal solutions for Towers of Hanoi problems."""
    
    def __init__(self):
        self.moves = []
    
    def solve(self, num_disks: int, source: int = 0, target: int = 2, 
              auxiliary: int = 1) -> List[Tuple[int, int]]:
        """
        Generate optimal solution using recursive algorithm.
        
        Returns:
            List of (from_peg, to_peg) tuples
        """
        self.moves = []
        self._hanoi_recursive(num_disks, source, target, auxiliary)
        return self.moves
    
    def _hanoi_recursive(self, n: int, source: int, target: int, auxiliary: int):
        """Recursive Towers of Hanoi solver."""
        if n == 1:
            self.moves.append((source, target))
        else:
            # Move n-1 disks from source to auxiliary
            self._hanoi_recursive(n - 1, source, auxiliary, target)
            # Move the largest disk from source to target
            self.moves.append((source, target))
            # Move n-1 disks from auxiliary to target
            self._hanoi_recursive(n - 1, auxiliary, target, source)
    
    def generate_reasoning_trace(self, num_disks: int, source: int = 0, 
                                  target: int = 2, auxiliary: int = 1) -> str:
        """
        Generate a detailed step-by-step reasoning trace with four layers:
        A. Subgoal Decomposition
        B. Precondition Checking
        C. The Action
        D. State Update (Post-condition)
        
        This is the "expert" trace used for supervised learning.
        """
        moves = self.solve(num_disks, source, target, auxiliary)
        state = TowersOfHanoiState(num_disks)
        
        trace = f"<think>\nSolving Towers of Hanoi with {num_disks} disks.\n"
        trace += f"Initial State: {state}\n"
        trace += f"Goal: Move all {num_disks} disks from peg {source} to peg {target}\n"
        trace += f"Available pegs: {source}, {auxiliary}, {target}\n\n"
        
        # Track subgoal hierarchy for recursive decomposition
        subgoal_stack = []
        if num_disks > 1:
            trace += "=== SUBGOAL DECOMPOSITION ===\n"
            trace += f"To move {num_disks} disks from {source} to {target}:\n"
            trace += f"1. First move {num_disks-1} smaller disks from {source} to {auxiliary} (using {target} as temporary)\n"
            trace += f"2. Then move disk {num_disks} from {source} to {target}\n"
            trace += f"3. Finally move {num_disks-1} smaller disks from {auxiliary} to {target} (using {source} as temporary)\n\n"
        
        for i, (from_peg, to_peg) in enumerate(moves, 1):
            disk = state.pegs[from_peg][-1]
            
            trace += f"{'='*60}\n"
            trace += f"STEP {i}/{len(moves)}\n"
            trace += f"{'='*60}\n\n"
            
            # A. SUBGOAL DECOMPOSITION
            trace += "A. SUBGOAL DECOMPOSITION:\n"
            if disk == num_disks:
                trace += f"   Main goal: Moving the largest disk (Disk {disk}) from {from_peg} to {to_peg}\n"
                trace += f"   This is the key move - all smaller disks must be out of the way\n"
            elif disk == 1:
                trace += f"   Moving the smallest disk (Disk {disk}) from {from_peg} to {to_peg}\n"
                trace += f"   Disk 1 can move freely to any peg\n"
            else:
                # Determine context
                disks_on_from = [d for d in state.pegs[from_peg] if d <= disk]
                if len(disks_on_from) == 1:
                    trace += f"   Subgoal: Move Disk {disk} from {from_peg} to {to_peg}\n"
                    trace += f"   This enables moving larger disks or repositioning disk groups\n"
                else:
                    trace += f"   Recursive subproblem: Moving Disk {disk} from {from_peg} to {to_peg}\n"
                    trace += f"   Part of moving {len(disks_on_from)} disks from {from_peg}\n"
            
            # B. PRECONDITION CHECKING
            trace += "\nB. PRECONDITION CHECKING:\n"
            trace += f"   Checking move: Disk {disk} from {from_peg} to {to_peg}\n"
            
            # Check source peg
            if not state.pegs[from_peg]:
                trace += f"   ✗ ERROR: Peg {from_peg} is empty!\n"
            else:
                top_disk = state.pegs[from_peg][-1]
                if top_disk == disk:
                    trace += f"   ✓ Disk {disk} is at the top of peg {from_peg}\n"
                else:
                    trace += f"   ✗ ERROR: Disk {disk} is not at the top (top is Disk {top_disk})\n"
            
            # Check destination peg
            if not state.pegs[to_peg]:
                trace += f"   ✓ Peg {to_peg} is empty - any disk can be placed here\n"
            else:
                top_dest = state.pegs[to_peg][-1]
                if disk < top_dest:
                    trace += f"   ✓ Disk {disk} (smaller) can be placed on Disk {top_dest} (larger)\n"
                else:
                    trace += f"   ✗ ERROR: Cannot place Disk {disk} on smaller Disk {top_dest}\n"
            
            # Overall validity
            is_valid = state.is_valid_move(from_peg, to_peg)
            trace += f"   {'✓' if is_valid else '✗'} Move is {'VALID' if is_valid else 'INVALID'}\n"
            
            # C. THE ACTION
            trace += "\nC. ACTION:\n"
            trace += f"   Executing: Move Disk {disk} from Peg {from_peg} to Peg {to_peg}\n"
            
            # Apply the move
            state.apply_move(from_peg, to_peg)
            
            # D. STATE UPDATE (POST-CONDITION)
            trace += "\nD. STATE UPDATE:\n"
            trace += f"   New State: {state}\n"
            
            # Additional state information
            remaining_on_source = len(state.pegs[source])
            completed_on_target = len(state.pegs[target])
            trace += f"   Progress: {completed_on_target}/{num_disks} disks on target peg {target}\n"
            
            if completed_on_target == num_disks:
                trace += f"   🎯 GOAL ACHIEVED!\n"
            
            trace += "\n"
        
        trace += "="*60 + "\n"
        trace += "SOLUTION COMPLETE\n"
        trace += "="*60 + "\n"
        trace += f"Final State: {state}\n"
        trace += f"Total moves: {len(moves)} (Optimal: {2**num_disks - 1})\n"
        trace += f"Goal: Move all disks from {source} to {target} - ✓ SUCCESS\n"
        trace += "</think>"
        
        return trace


class TowersOfHanoiValidator:
    """
    Validates Towers of Hanoi reasoning traces and computes rewards.
    Extends the base PlanningValidator for the Towers of Hanoi domain.
    """
    
    def __init__(self):
        self.domain_name = "towers_of_hanoi"
        # Updated patterns to match the new trace format
        self.move_pattern = re.compile(
            r'(?:move|Move|Executing:\s*Move)\s+(?:disk\s+)?(\d+)\s+(?:from\s+)?(?:peg\s+)?([ABC])\s+(?:to\s+)?(?:peg\s+)?([ABC])',
            re.IGNORECASE
        )
    
    def parse_moves(self, reasoning_trace: str) -> List[List[int]]:
        """
        Extract moves from the final answer (outside <think> tags).
        
        DeepSeek R1 models use <think>...</think> for reasoning. The final answer
        with moves = [[disk id, from peg, to peg], ...] should appear AFTER </think>.
        
        Returns:
            List of [disk, from_peg, to_peg] moves (all integers, 0-indexed)
        """
        import json
        
        moves_pattern = r'moves\s*=\s*(\[(?:\s*\[[^\]]+\]\s*,?\s*)+\])'
        
        # Remove content inside <think>...</think> tags to get only the final answer
        text_outside_think = re.sub(r'<think>.*?</think>', '', reasoning_trace, flags=re.DOTALL)
        
        # Look for moves = [[...]] in the final answer (outside think tags)
        candidates = re.findall(moves_pattern, text_outside_think, re.DOTALL)
        
        if len(candidates) == 0:
            raise ValueError(
                "No moves found in final answer (outside <think> tags). "
                "Expected format: moves = [[disk id, from peg, to peg], ...]"
            )
        
        # Take the last occurrence in case there are multiple
        moves_str = candidates[-1].strip()
        # Remove any trailing text after the final ]
        moves_str = re.sub(r'\]\s*[^\]]*$', ']', moves_str)
        
        try:
            moves = json.loads(moves_str)
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse moves array: {e}")
        
        return moves
    
    def validate_trace(self, reasoning_trace: str, problem_state: Dict) -> Tuple[float, int]:
        """
        Validate a reasoning trace for Towers of Hanoi.
        Follows the same validation logic as the reference implementation.
        
        Args:
            reasoning_trace: Generated reasoning text
            problem_state: Dict with 'num_disks', 'goal_peg' (default 2)
        
        Returns:
            reward: float - Reward score (1.0+ if solved, 0.0 otherwise)
            violations: int - Number of constraint violations (0 if successful)
        """
        num_disks = problem_state.get('num_disks', 3)
        goal_peg = problem_state.get('goal_peg', 2)
        
        # Initialize state
        state = TowersOfHanoiState(num_disks)
        goal_state = [[], [], list(range(num_disks, 0, -1))]
        
        try:
            # Parse moves from trace
            moves = self.parse_moves(reasoning_trace)
        except ValueError as e:
            # Failed to parse - return 0 reward and 1 violation
            return 0.0, 1
        
        violations = 0
        
        try:
            # Validate and play each move
            for move in moves:
                if len(move) != 3 or not all(isinstance(x, int) for x in move):
                    raise ValueError(f"Invalid move: {move}. Must be a list of three integers.")
                
                disk, from_peg, to_peg = move
                
                # Validate disk number
                if disk > num_disks or disk < 1:
                    raise ValueError(f"Invalid disk number: {disk}. Must be between 1 and {num_disks}")
                
                # Validate peg numbers
                if from_peg < 0 or from_peg > 2:
                    raise ValueError(f"Invalid from peg number: {from_peg}. Must be between 0 and 2")
                if to_peg < 0 or to_peg > 2:
                    raise ValueError(f"Invalid to peg number: {to_peg}. Must be between 0 and 2")
                
                # Validate the disk is on top of from_peg
                if len(state.pegs[from_peg]) == 0 or state.pegs[from_peg][-1] != disk:
                    raise ValueError(f"From peg {from_peg} does not contain disk {disk}")
                
                # Validate the move is legal (no larger disk on smaller)
                if len(state.pegs[to_peg]) > 0 and state.pegs[to_peg][-1] < disk:
                    raise ValueError(f"Cannot place disk {disk} on top of disk {state.pegs[to_peg][-1]}")
                
                # Execute the move
                state.pegs[to_peg].append(state.pegs[from_peg].pop())
        
        except ValueError as e:
            # Invalid move encountered - count as violation
            violations += 1
        
        # Check if solved
        solved = state.pegs == goal_state
        
        if solved:
            # Compute reward based on optimality
            optimal_moves = 2 ** num_disks - 1
            if len(moves) == optimal_moves:
                reward = 1.5  # Bonus for optimal solution
            elif len(moves) <= optimal_moves * 1.2:  # Within 20% of optimal
                reward = 1.2
            else:
                reward = 1.0
        else:
            reward = 0.0
        
        return reward, violations
    
    def batch_validate(self, traces: List[str], problem_states: List[Dict]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Validate a batch of reasoning traces.
        
        Returns:
            rewards: (batch_size,) tensor
            violations: (batch_size,) tensor
        """
        rewards = []
        violations = []
        
        for trace, state in zip(traces, problem_states):
            r, v = self.validate_trace(trace, state)
            rewards.append(r)
            violations.append(v)
        
        return torch.tensor(rewards), torch.tensor(violations)


class NonStandardValidator:
    """
    Validates solutions for non-standard TOH configurations.
    
    Unlike the standard validator, this handles arbitrary initial and goal states,
    not just the canonical "all disks on peg 0 -> all disks on peg 2" configuration.
    """
    
    def __init__(self):
        self.move_pattern = re.compile(
            r'moves\s*=\s*(\[(?:\s*\[[^\]]+\]\s*,?\s*)+\])',
            re.DOTALL
        )
    
    def parse_moves(self, response: str) -> Optional[List[List[int]]]:
        """Extract moves array from response."""
        import json
        matches = self.move_pattern.findall(response)
        if not matches:
            return None
        
        moves_str = matches[-1].strip()
        try:
            moves = json.loads(moves_str)
            return moves
        except json.JSONDecodeError:
            return None
    
    def validate(
        self, 
        response: str, 
        initial_state: List[List[int]], 
        goal_state: List[List[int]],
        num_disks: int
    ) -> Dict:
        """
        Validate a solution for non-standard configuration.
        
        Args:
            response: Model's response containing moves
            initial_state: Starting configuration [[peg0], [peg1], [peg2]]
            goal_state: Target configuration [[peg0], [peg1], [peg2]]
            num_disks: Number of disks in the puzzle
        
        Returns:
            Dict with validation results:
                - success: bool (whether parsing succeeded)
                - violations: int (number of rule violations)
                - num_moves: int (number of moves in solution)
                - solved: bool (whether goal state was reached)
                - final_state: List[List[int]] (state after applying moves)
                - error: str (error message if parsing failed)
        """
        moves = self.parse_moves(response)
        
        if moves is None:
            return {
                'success': False,
                'error': 'Failed to parse moves',
                'violations': 1,
                'num_moves': 0,
                'solved': False,
            }
        
        # Simulate the moves
        state = [peg.copy() for peg in initial_state]
        violations = 0
        
        for i, move in enumerate(moves):
            if len(move) != 3:
                violations += 1
                continue
            
            disk, from_peg, to_peg = move
            
            # Validate disk number
            if disk < 1 or disk > num_disks:
                violations += 1
                continue
            
            # Validate peg numbers
            if from_peg < 0 or from_peg > 2 or to_peg < 0 or to_peg > 2:
                violations += 1
                continue
            
            # Check source peg has the disk on top
            if not state[from_peg] or state[from_peg][-1] != disk:
                violations += 1
                continue
            
            # Check destination peg constraint
            if state[to_peg] and state[to_peg][-1] < disk:
                violations += 1
                continue
            
            # Apply the move
            state[from_peg].pop()
            state[to_peg].append(disk)
        
        # Check if goal reached
        solved = state == goal_state
        
        return {
            'success': True,
            'violations': violations,
            'num_moves': len(moves),
            'solved': solved,
            'final_state': state,
        }


class TowersOfHanoiDataset:
    """Generates Towers of Hanoi problems and expert solutions."""
    
    def __init__(self, min_disks: int = 2, max_disks: int = 5):
        self.min_disks = min_disks
        self.max_disks = max_disks
        self.solver = TowersOfHanoiSolver()
    
    def generate_problem(self, num_disks: Optional[int] = None) -> Dict:
        """
        Generate a single Towers of Hanoi problem.
        
        Returns:
            Dict with:
                - 'num_disks': int
                - 'initial_state': TowersOfHanoiState
                - 'goal_peg': str
                - 'problem_text': str (natural language description)
                - 'expert_trace': str (optimal solution with reasoning)
                - 'optimal_moves': List[Tuple[str, str]]
        """
        if num_disks is None:
            import random
            num_disks = random.randint(self.min_disks, self.max_disks)
        
        initial_state = TowersOfHanoiState(num_disks)
        goal_peg = 2
        
        problem_text = f"""Solve the Towers of Hanoi puzzle with {num_disks} disks.
Initial state: All {num_disks} disks are on peg 0 (disk 1 is smallest, disk {num_disks} is largest).
Goal: Move all disks to peg {goal_peg}.
Rules: 
1. Only one disk can be moved at a time.
2. A disk can only be placed on top of a larger disk or on an empty peg.
3. Only the top disk of a peg can be moved.

Please provide the sequence of moves to solve this puzzle."""
        
        expert_trace = self.solver.generate_reasoning_trace(num_disks)
        optimal_moves = self.solver.solve(num_disks)
        
        return {
            'num_disks': num_disks,
            'initial_state': initial_state,
            'goal_peg': goal_peg,
            'problem_text': problem_text,
            'expert_trace': expert_trace,
            'optimal_moves': optimal_moves
        }
    
    def generate_batch(self, batch_size: int) -> List[Dict]:
        """Generate a batch of problems."""
        return [self.generate_problem() for _ in range(batch_size)]


# ============================================================================
# Move Parser and Constraint Checker
# ============================================================================

class MoveParser:
    """Parses moves from reasoning traces and checks constraints."""
    
    def __init__(self):
        # Pattern for extracting individual moves during reasoning
        # Matches patterns like: "Move disk 1 from peg 0 to peg 2"
        self.move_patterns = [
            # Pattern: "Move disk X from peg Y to peg Z"
            re.compile(
                r'[Mm]ove\s+[Dd]isk\s+(\d+)\s+from\s+[Pp]eg\s+(\d+)\s+to\s+[Pp]eg\s+(\d+)',
                re.IGNORECASE
            ),
            # Pattern: "[disk, from, to]" or "[X, Y, Z]"
            re.compile(r'\[\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\]'),
            # Pattern: "disk X: Y -> Z" or "disk X from Y to Z"
            re.compile(r'[Dd]isk\s+(\d+)[:\s]+(\d+)\s*(?:->|→|to)\s*(\d+)'),
        ]
        
        # Pattern for final moves array
        self.final_moves_pattern = re.compile(
            r'moves\s*=\s*(\[(?:\s*\[[^\]]+\]\s*,?\s*)+\])',
            re.DOTALL
        )
    
    def parse_final_moves(self, text: str) -> Optional[List[List[int]]]:
        """
        Parse the final moves array from the solution (outside <think> tags).
        
        DeepSeek R1 models output reasoning in <think>...</think> tags.
        The final answer should appear AFTER </think>.
        
        Returns:
            List of [disk, from_peg, to_peg] or None if not found
        """
        # Remove content inside <think>...</think> tags
        text_outside_think = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
        
        # Look for moves in the final answer only
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
        """
        Parse all moves mentioned in the reasoning trace.
        
        Returns:
            List of (position_in_text, [disk, from_peg, to_peg])
        """
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
        
        # Sort by position in text
        moves.sort(key=lambda x: x[0])
        return moves
    
    def extract_reasoning_moves(self, text: str) -> List[List[int]]:
        """
        Extract all moves from reasoning (thinking) section.
        
        Returns:
            Ordered list of moves found in reasoning
        """
        # Extract thinking section if present
        think_match = re.search(r'<think>(.*?)</think>', text, re.DOTALL)
        if think_match:
            reasoning = think_match.group(1)
        else:
            reasoning = text
        
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
        step_index: int
    ) -> List[MoveViolation]:
        """
        Check a single move for constraint violations.
        
        Args:
            move: [disk, from_peg, to_peg]
            state: Current state before the move
            step_index: Index of this move in the sequence
            
        Returns:
            List of violations found
        """
        violations = []
        
        # Check move format
        if len(move) != 3:
            violations.append(MoveViolation(
                violation_type=ViolationType.INVALID_MOVE_FORMAT,
                move=move,
                step_index=step_index,
                description=f"Move must have 3 elements, got {len(move)}",
                severity=1.0
            ))
            return violations
        
        disk, from_peg, to_peg = move
        
        # Check disk number validity
        if not isinstance(disk, int) or disk < 1 or disk > self.num_disks:
            violations.append(MoveViolation(
                violation_type=ViolationType.INVALID_DISK_NUMBER,
                move=move,
                step_index=step_index,
                description=f"Invalid disk number {disk}, must be 1-{self.num_disks}",
                severity=1.0
            ))
            return violations
        
        # Check peg number validity
        if not isinstance(from_peg, int) or from_peg < 0 or from_peg > 2:
            violations.append(MoveViolation(
                violation_type=ViolationType.INVALID_PEG_NUMBER,
                move=move,
                step_index=step_index,
                description=f"Invalid source peg {from_peg}, must be 0-2",
                severity=1.0
            ))
            return violations
        
        if not isinstance(to_peg, int) or to_peg < 0 or to_peg > 2:
            violations.append(MoveViolation(
                violation_type=ViolationType.INVALID_PEG_NUMBER,
                move=move,
                step_index=step_index,
                description=f"Invalid destination peg {to_peg}, must be 0-2",
                severity=1.0
            ))
            return violations
        
        # Check source peg is not empty
        if len(state.pegs[from_peg]) == 0:
            violations.append(MoveViolation(
                violation_type=ViolationType.SOURCE_PEG_EMPTY,
                move=move,
                step_index=step_index,
                description=f"Source peg {from_peg} is empty",
                severity=1.0
            ))
            return violations
        
        # Check the disk is on top of the source peg
        top_disk = state.pegs[from_peg][-1]
        if top_disk != disk:
            violations.append(MoveViolation(
                violation_type=ViolationType.DISK_NOT_ON_TOP,
                move=move,
                step_index=step_index,
                description=f"Disk {disk} is not on top of peg {from_peg}, top disk is {top_disk}",
                severity=1.0
            ))
            return violations
        
        # Check destination peg constraint (larger disk on smaller)
        if len(state.pegs[to_peg]) > 0:
            top_dest = state.pegs[to_peg][-1]
            if disk > top_dest:
                violations.append(MoveViolation(
                    violation_type=ViolationType.LARGER_ON_SMALLER,
                    move=move,
                    step_index=step_index,
                    description=f"Cannot place disk {disk} on smaller disk {top_dest}",
                    severity=1.0
                ))
        
        return violations
    
    def check_move_sequence(
        self, 
        moves: List[List[int]]
    ) -> Tuple[List[MoveViolation], TowersOfHanoiState]:
        """
        Check a sequence of moves for constraint violations.
        
        Args:
            moves: List of [disk, from_peg, to_peg] moves
            
        Returns:
            Tuple of (all_violations, final_state)
        """
        state = self.initial_state.copy()
        all_violations = []
        
        for i, move in enumerate(moves):
            violations = self.check_move(move, state, i)
            all_violations.extend(violations)
            
            # Apply move if valid (for state tracking)
            if not violations and len(move) == 3:
                disk, from_peg, to_peg = move
                if (len(state.pegs[from_peg]) > 0 and 
                    state.pegs[from_peg][-1] == disk):
                    state.pegs[from_peg].pop()
                    state.pegs[to_peg].append(disk)
        
        return all_violations, state
    
    def compute_violation_score(self, violations: List[MoveViolation]) -> float:
        """
        Compute a normalized violation score.
        
        Returns:
            Score in [0, 1] where 0 = no violations, 1 = many violations
        """
        if not violations:
            return 0.0
        
        total_severity = sum(v.severity for v in violations)
        # Normalize by expected number of moves (2^n - 1)
        expected_moves = 2 ** self.num_disks - 1
        normalized = total_severity / max(expected_moves, 1)
        return min(normalized, 1.0)


# ============================================================================
# GRPO Training Dataset
# ============================================================================

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
        
        # Pre-generate problems
        self.problems = self._generate_problems()
    
    def _generate_problems(self) -> List[Dict]:
        """Generate problems with equal proportions of each disk count."""
        problems = []
        dataset = TowersOfHanoiDataset(self.min_disks, self.max_disks)
        
        # Equal distribution across disk counts
        disk_counts = list(range(self.min_disks, self.max_disks + 1))
        num_per_disk = self.num_problems // len(disk_counts)
        remainder = self.num_problems % len(disk_counts)
        
        for i, num_disks in enumerate(disk_counts):
            # Distribute remainder across first few disk counts
            count = num_per_disk + (1 if i < remainder else 0)
            
            for _ in range(count):
                problem = dataset.generate_problem(num_disks=num_disks)
                system_prompt, user_prompt = create_standard_prompt(num_disks)
                
                problems.append({
                    'num_disks': num_disks,
                    'goal_peg': 2,
                    'system_prompt': system_prompt,
                    'user_prompt': user_prompt,
                    'optimal_moves': 2 ** num_disks - 1,
                })
        
        # Shuffle to avoid ordered training
        random.shuffle(problems)
        return problems
    
    def __len__(self) -> int:
        return self.num_problems
    
    def __getitem__(self, idx: int) -> Dict:
        return self.problems[idx]


if __name__ == "__main__":
    # Test the validator and solver
    print("=" * 60)
    print("Testing Towers of Hanoi Validator and Solver")
    print("=" * 60)
    
    # Test solver
    solver = TowersOfHanoiSolver()
    for n in [2, 3, 4]:
        moves = solver.solve(n)
        print(f"\nOptimal solution for {n} disks ({len(moves)} moves):")
        print(moves)
    
    # Test reasoning trace generation
    print("\n" + "=" * 60)
    print("Expert Reasoning Trace (3 disks):")
    print("=" * 60)
    trace = solver.generate_reasoning_trace(3)
    print(trace)
    
    # Test validator
    print("\n" + "=" * 60)
    print("Testing Validator:")
    print("=" * 60)
    validator = TowersOfHanoiValidator()
    
    problem_state = {'num_disks': 3, 'goal_peg': 'C'}
    reward, violations = validator.validate_trace(trace, problem_state)
    print(f"\nReward: {reward}, Violations: {violations}")
    
    # Test with incorrect trace
    bad_trace = """<think>
    Move disk 3 from A to C
    Move disk 2 from A to B
    Move disk 3 from C to B
    </think>"""
    
    reward, violations = validator.validate_trace(bad_trace, problem_state)
    print(f"\nBad trace - Reward: {reward}, Violations: {violations}")
    
    # Test dataset generation
    print("\n" + "=" * 60)
    print("Testing Dataset Generation:")
    print("=" * 60)
    dataset = TowersOfHanoiDataset(min_disks=2, max_disks=4)
    problem = dataset.generate_problem(num_disks=3)
    print(f"\nProblem Text:\n{problem['problem_text']}")
    print(f"\nNumber of optimal moves: {len(problem['optimal_moves'])}")
