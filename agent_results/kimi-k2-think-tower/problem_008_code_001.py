# Tool: write_file
# Target path: /home/user/hanoi_solver.py
# Step: 2

#!/usr/bin/env python3
"""
Tower of Hanoi solver for 3 disks.
Initial: Peg 0 has [3, 2, 1], Pegs 1 and 2 are empty
Goal: Peg 2 has [3, 2, 1], Pegs 0 and 1 are empty
"""

def apply_move(state, move):
    """Apply a move to the state and return new state."""
    disk, from_peg, to_peg = move
    new_state = [peg.copy() for peg in state]
    
    # Remove disk from source peg
    if not new_state[from_peg] or new_state[from_peg][-1] != disk:
        raise ValueError(f"Invalid move: disk {disk} not on top of peg {from_peg}")
    new_state[from_peg].pop()
    
    # Add disk to destination peg
    if new_state[to_peg] and new_state[to_peg][-1] < disk:
        raise ValueError(f"Invalid move: cannot place disk {disk} on smaller disk")
    new_state[to_peg].append(disk)
    
    return new_state

def print_state(state, step):
    """Print the current state of the pegs."""
    print(f"\nStep {step}:")
    for i, peg in enumerate(state):
        print(f"  Peg {i}: {peg}")

def solve_hanoi_3_disks():
    """Solve the Tower of Hanoi with 3 disks."""
    # Initial state: Peg 0 has [3,2,1], others empty
    initial_state = [[3, 2, 1], [], []]
    
    # Goal state: Peg 2 has [3,2,1], others empty
    goal_state = [[], [], [3, 2, 1]]
    
    # Optimal sequence of moves for 3 disks
    moves = [
        [1, 0, 2],  # Move disk 1 from peg 0 to peg 2
        [2, 0, 1],  # Move disk 2 from peg 0 to peg 1
        [1, 2, 1],  # Move disk 1 from peg 2 to peg 1
        [3, 0, 2],  # Move disk 3 from peg 0 to peg 2
        [1, 1, 0],  # Move disk 1 from peg 1 to peg 0
        [2, 1, 2],  # Move disk 2 from peg 1 to peg 2
        [1, 0, 2],  # Move disk 1 from peg 0 to peg 2
    ]
    
    print("Tower of Hanoi Solution for 3 disks")
    print("=" * 50)
    print("\nInitial state:")
    print_state(initial_state, 0)
    
    current_state = initial_state
    
    for i, move in enumerate(moves, 1):
        disk, from_peg, to_peg = move
        print(f"\nMove {i}: Move disk {disk} from peg {from_peg} to peg {to_peg}")
        
        current_state = apply_move(current_state, move)
        print_state(current_state, i)
    
    # Verify final state
    if current_state == goal_state:
        print("\n" + "=" * 50)
        print("✓ SUCCESS! Goal state reached!")
        print("=" * 50)
        return True, moves
    else:
        print("\n" + "=" * 50)
        print("✗ FAILED! Did not reach goal state.")
        print("=" * 50)
        return False, moves

if __name__ == "__main__":
    success, moves = solve_hanoi_3_disks()
    if success:
        print(f"\nOptimal sequence of {len(moves)} moves:")
        print(f"moves = {moves}")
    else:
        print("\nSolution verification failed.")