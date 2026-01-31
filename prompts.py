"""
Shared prompt templates and generation functions for Towers of Hanoi.

This module contains all prompt-related code used across baseline.py, train.py, and test.py
to avoid duplication.
"""

import random
from typing import Dict, List, Tuple


# ============================================================================
# System Prompt (Constant)
# ============================================================================

SYSTEM_PROMPT = """You are a helpful assistant. Solve this puzzle for me.
There are three pegs and n disks of different sizes stacked on the first peg.
The disks are numbered from 1 (smallest) to n (largest). Disk moves in this puzzle should follow:
1. Only one disk can be moved at a time.
2. Each move consists of taking the upper disk from one stack and placing it on top of another stack.
3. A larger disk may not be placed on top of a smaller disk.
The goal is to move the entire stack to the third peg.

Example: With 3 disks numbered 1 (smallest), 2, and 3 (largest), the initial state is [[3, 2, 1], [], []], and a solution might be:

moves = [[1 , 0, 2], [2, 0, 1], [1, 2, 1], [3, 0, 2], [1, 1, 0], [2, 1, 2], [1, 0, 2]]

This means: Move disk 1 from peg 0 to peg 2, then move disk 2 from peg 0 to peg 1, and so on.

Requirements:
• When exploring potential solutions in your thinking process, always include the corresponding
complete list of moves.
• The positions are 0-indexed (the leftmost peg is 0).
• Ensure your final answer includes the complete list of moves in the format: moves = [[disk id, from peg, to peg], ...]"""


# ============================================================================
# Helper Functions
# ============================================================================

def format_peg_state(peg: List[int]) -> str:
    """
    Format a peg's disk stack for the prompt.
    
    Args:
        peg: List of disks on the peg (bottom to top)
        
    Returns:
        Formatted string representation
    """
    if not peg:
        return "(empty)"
    
    if len(peg) == 1:
        return f"{peg[0]} (top)"
    
    # peg is stored bottom to top: [3, 2, 1] means 3 at bottom, 1 at top
    parts = [f"{peg[0]} (bottom)"]
    if len(peg) > 2:
        parts.extend(str(d) for d in peg[1:-1])
    parts.append(f"{peg[-1]} (top)")
    return ", ".join(parts)


# ============================================================================
# Standard TOH Prompt Creation
# ============================================================================

def create_standard_prompt(num_disks: int, goal_peg: int = 2) -> Tuple[str, str]:
    """
    Create standard TOH prompt: all disks start on peg 0, goal is specified peg.
    
    Args:
        num_disks: Number of disks in the puzzle
        goal_peg: Target peg (default 2)
        
    Returns:
        Tuple of (system_prompt, user_prompt)
    """
    # Generate the state strings for standard configuration
    if num_disks == 1:
        state = "1 (top)"
    else:
        stack = list(range(num_disks, 0, -1))
        state = ", ".join([f"{stack[0]} (bottom)"] + list(map(str, stack[1:-1]))) + f", {stack[-1]} (top)"
    
    user_prompt = f"""I have a puzzle with {num_disks} disks of different sizes with
Initial configuration:
• Peg 0: {state}
• Peg 1: (empty)
• Peg 2: (empty)

Goal configuration:
• Peg 0: (empty)
• Peg 1: (empty)
• Peg 2: {state}

Rules:
• Only one disk can be moved at a time.
• Only the top disk from any stack can be moved.
• A larger disk may not be placed on top of a smaller disk.

Find the sequence of moves to transform the initial configuration into the goal configuration."""
    
    return SYSTEM_PROMPT, user_prompt


def create_standard_prompt_with_info(num_disks: int) -> Tuple[str, str, Dict]:
    """
    Create standard TOH prompt with problem info dict (for test.py).
    
    Returns:
        Tuple of (system_prompt, user_prompt, problem_info)
    """
    # Standard initial state: all disks on peg 0
    initial_state = [list(range(num_disks, 0, -1)), [], []]
    # Standard goal state: all disks on peg 2
    goal_state = [[], [], list(range(num_disks, 0, -1))]
    
    initial_peg0 = format_peg_state(initial_state[0])
    initial_peg1 = format_peg_state(initial_state[1])
    initial_peg2 = format_peg_state(initial_state[2])
    
    goal_peg0 = format_peg_state(goal_state[0])
    goal_peg1 = format_peg_state(goal_state[1])
    goal_peg2 = format_peg_state(goal_state[2])
    
    user_prompt = f"""I have a puzzle with {num_disks} disks of different sizes with
Initial configuration:
• Peg 0: {initial_peg0}
• Peg 1: {initial_peg1}
• Peg 2: {initial_peg2}

Goal configuration:
• Peg 0: {goal_peg0}
• Peg 1: {goal_peg1}
• Peg 2: {goal_peg2}

Rules:
• Only one disk can be moved at a time.
• Only the top disk from any stack can be moved.
• A larger disk may not be placed on top of a smaller disk.

Find the sequence of moves to transform the initial configuration into the goal configuration."""
    
    problem_info = {
        'num_disks': num_disks,
        'initial_state': initial_state,
        'goal_state': goal_state,
        'config_type': 'standard',
    }
    
    return SYSTEM_PROMPT, user_prompt, problem_info


# ============================================================================
# Non-standard TOH Prompt Creation
# ============================================================================

def generate_valid_toh_state(num_disks: int, rng: random.Random) -> List[List[int]]:
    """
    Generate a valid TOH state by randomly distributing disks across pegs.
    Each disk is placed on a random peg, maintaining the constraint that
    larger disks are always below smaller disks on any peg.
    
    Args:
        num_disks: Number of disks
        rng: Random number generator for reproducibility
        
    Returns:
        List of 3 lists representing the pegs (bottom to top ordering)
    """
    pegs = [[], [], []]
    
    # Place disks from largest to smallest (this ensures validity)
    for disk in range(num_disks, 0, -1):
        peg_idx = rng.randint(0, 2)
        pegs[peg_idx].append(disk)
    
    return pegs


def states_are_equal(state1: List[List[int]], state2: List[List[int]]) -> bool:
    """Check if two TOH states are equal."""
    return state1 == state2


def create_nonstandard_prompt(
    num_disks: int, 
    problem_id: int,
    seed: int
) -> Tuple[str, str, Dict]:
    """
    Create non-standard TOH prompt with random start and goal configurations.
    
    Args:
        num_disks: Number of disks
        problem_id: Unique identifier for this problem (for reproducibility)
        seed: Base random seed
        
    Returns:
        Tuple of (system_prompt, user_prompt, problem_info)
    """
    # Create deterministic RNG based on seed, num_disks, and problem_id
    problem_seed = seed + num_disks * 1000 + problem_id
    rng = random.Random(problem_seed)
    
    # Generate random initial and goal states
    # Keep generating until they're different
    initial_state = generate_valid_toh_state(num_disks, rng)
    goal_state = generate_valid_toh_state(num_disks, rng)
    
    # Ensure initial and goal states are different
    max_attempts = 100
    attempts = 0
    while states_are_equal(initial_state, goal_state) and attempts < max_attempts:
        goal_state = generate_valid_toh_state(num_disks, rng)
        attempts += 1
    
    initial_peg0 = format_peg_state(initial_state[0])
    initial_peg1 = format_peg_state(initial_state[1])
    initial_peg2 = format_peg_state(initial_state[2])
    
    goal_peg0 = format_peg_state(goal_state[0])
    goal_peg1 = format_peg_state(goal_state[1])
    goal_peg2 = format_peg_state(goal_state[2])
    
    user_prompt = f"""I have a puzzle with {num_disks} disks of different sizes with
Initial configuration:
• Peg 0: {initial_peg0}
• Peg 1: {initial_peg1}
• Peg 2: {initial_peg2}

Goal configuration:
• Peg 0: {goal_peg0}
• Peg 1: {goal_peg1}
• Peg 2: {goal_peg2}

Rules:
• Only one disk can be moved at a time.
• Only the top disk from any stack can be moved.
• A larger disk may not be placed on top of a smaller disk.

Find the sequence of moves to transform the initial configuration into the goal configuration."""
    
    problem_info = {
        'num_disks': num_disks,
        'initial_state': initial_state,
        'goal_state': goal_state,
        'config_type': 'nonstandard',
        'problem_id': problem_id,
        'problem_seed': problem_seed,
    }
    
    return SYSTEM_PROMPT, user_prompt, problem_info
