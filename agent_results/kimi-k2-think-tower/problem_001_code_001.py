# Tool: write_file
# Target path: /home/user/hanoi_solver.py
# Step: 2

#!/usr/bin/env python3
"""
Tower of Hanoi solver for 3 disks.
Verifies the optimal sequence of moves from peg 0 to peg 2.
"""

def apply_move(state, move):
    """Apply a single move to the state."""
    disk, from_peg, to_peg = move
    
    # Check that the disk is at the top of from_peg
    if not state[from_peg] or state[from_peg][-1] != disk:
        raise ValueError(f"Invalid move: disk {disk} is not at top of peg {from_peg}")
    
    # Check that we can place disk on to_peg (only smaller disks allowed below)
    if state[to_peg] and state[to_peg][-1] < disk:
        raise ValueError(f"Invalid move: cannot place disk {disk} on smaller disk {state[to_peg][-1]}")
    
    # Perform the move
    state[from_peg].pop()
    state[to_peg].append(disk)
    
    return state

def print_state(state, step):
    """Print the current state."""
    print(f"Step {step}:")
    for i, peg in enumerate(state):
        print(f"  Peg {i}: {peg}")
    print()

def solve_hanoi_3_disks():
    """Solve the Tower of Hanoi with 3 disks."""
    # Initial state: peg 0 has [3, 2, 1], others are empty
    initial_state = [[3, 2, 1], [], []]
    
    # Goal state: peg 2 has [3, 2, 1], others are empty
    goal_state = [[], [], [3, 2, 1]]
    
    # Optimal sequence of moves for 3 disks (7 moves)
    moves = [
        [1, 0, 2],  # Move disk 1 from peg 0 to peg 2
        [2, 0, 1],  # Move disk 2 from peg 0 to peg 1
        [1, 2, 1],  # Move disk 1 from peg 2 to peg 1
        [3, 0, 2],  # Move disk 3 from peg 0 to peg 2
        [1, 1, 0],  # Move disk 1 from peg 1 to peg 0
        [2, 1, 2],  # Move disk 2 from peg 1 to peg 2
        [1, 0, 2],  # Move disk 1 from peg 0 to peg 2
    ]
    
    print("Tower of Hanoi - 3 Disks")
    print("=" * 50)
    print("Initial state:")
    print_state(initial_state, 0)
    
    state = [peg.copy() for peg in initial_state]
    
    for i, move in enumerate(moves, 1):
        disk, from_peg, to_peg = move
        print(f"Move {i}: Disk {disk} from peg {from_peg} to peg {to_peg}")
        
        state = apply_move(state, move)
        print_state(state, i)
    
    # Verify final state
    if state == goal_state:
        print("✓ SUCCESS! Final state matches goal configuration.")
        print(f"✓ Solved in {len(moves)} moves (optimal for 3 disks)")
    else:
        print("✗ FAILED! Final state does not match goal.")
        print(f"Expected: {goal_state}")
        print(f"Got:      {state}")
    
    return moves

if __name__ == "__main__":
    moves = solve_hanoi_3_disks()
    print("\n" + "=" * 50)
    print("Complete list of moves:")
    print(f"moves = {moves}")