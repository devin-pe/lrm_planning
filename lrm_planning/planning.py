"""
Towers of Hanoi Planning Validator and Expert Solver

This module implements:
1. TowersOfHanoiValidator: Validates reasoning traces and computes rewards
2. TowersOfHanoiSolver: Generates optimal solutions for supervision
3. State tracking and constraint checking for the Towers of Hanoi domain
"""

import re
from typing import List, Dict, Tuple, Optional
import torch
from copy import deepcopy


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
        Extract moves from reasoning trace.
        Follows the same pattern as reference implementation.
        
        Returns:
            List of [disk, from_peg, to_peg] moves (all integers, 0-indexed)
        """
        import json
        
        # Find the moves array in the format: moves = [[disk, from, to], ...]
        candidates = re.findall(r'moves\s*=\s*(\[\[.*?\]\])', reasoning_trace, re.DOTALL)
        
        if len(candidates) == 0:
            raise ValueError(
                "No moves found in solution. Expected format: moves = [[disk id, from peg, to peg], ...]"
            )
        
        # Parse the JSON array (take the last occurrence)
        moves = json.loads(candidates[-1])
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
