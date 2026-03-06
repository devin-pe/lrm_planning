# Tool: write_file
# Target path: /home/user/hanoi_solver.py
# Step: 2

#!/usr/bin/env python3
"""
Tower of Hanoi solver for 3 disks.
Verifies the sequence of moves from initial to goal configuration.
"""

def apply_move(state, move):
    """Apply a single move to the state."""
    disk, from_peg, to_peg = move
    
    # Check that the from_peg has the disk
    if not state[from_peg] or state[from_peg][-1] != disk:
        raise ValueError(f"Invalid move: disk {disk} not on top of peg {from_peg}")
    
    # Check that we can place the disk on to_peg (no smaller disk there)
    if state[to_peg] and state[to_peg][-1] < disk:
        raise ValueError(f"Invalid move: cannot place disk {disk} on top of smaller disk {state[to_peg][-1]}")
    
    # Make the move
    state[from_peg].pop()
    state[to_peg].append(disk)
    
    return state

def simulate_moves(initial_state, moves):
    """Simulate a sequence of moves and return the final state."""
    state = [peg.copy() for peg in initial_state]
    
    print("Initial state:")
    print(f"  Peg 0: {state[0]}")
    print(f"  Peg 1: {state[1]}")
    print(f"  Peg 2: {state[2]}")
    print()
    
    for i, move in enumerate(moves, 1):
        disk, from_peg, to_peg = move
        print(f"Move {i}: Disk {disk} from peg {from_peg} to peg {to_peg}")
        state = apply_move(state, move)
        print(f"  Peg 0: {state[0]}")
        print(f"  Peg 1: {state[1]}")
        print(f"  Peg 2: {state[2]}")
        print()
    
    return state

def main():
    # Initial configuration
    initial_state = [[3, 2, 1], [], []]
    
    # Goal configuration
    goal_state = [[], [], [3, 2, 1]]
    
    # Sequence of moves (optimal solution for 3 disks)
    moves = [
        [1, 0, 2],  # Move disk 1 from peg 0 to peg 2
        [2, 0, 1],  # Move disk 2 from peg 0 to peg 1
        [1, 2, 1],  # Move disk 1 from peg 2 to peg 1
        [3, 0, 2],  # Move disk 3 from peg 0 to peg 2
        [1, 1, 0],  # Move disk 1 from peg 1 to peg 0
        [2, 1, 2],  # Move disk 2 from peg 1 to peg 2
        [1, 0, 2],  # Move disk 1 from peg 0 to peg 2
    ]
    
    print("Tower of Hanoi - 3 disks")
    print("=" * 50)
    print()
    
    final_state = simulate_moves(initial_state, moves)
    
    print("Final state:")
    print(f"  Peg 0: {final_state[0]}")
    print(f"  Peg 1: {final_state[1]}")
    print(f"  Peg 2: {final_state[2]}")
    print()
    
    # Verify if we reached the goal
    if final_state == goal_state:
        print("✓ SUCCESS! Reached goal configuration!")
        print()
        print("Complete list of moves:")
        print(f"moves = {moves}")
    else:
        print("✗ FAILED! Did not reach goal configuration.")
        print(f"Expected: {goal_state}")
        print(f"Got:      {final_state}")

if __name__ == "__main__":
    main()