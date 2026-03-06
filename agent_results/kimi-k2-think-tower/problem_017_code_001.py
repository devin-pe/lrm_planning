# Tool: write_file
# Target path: /home/user/hanoi_solver.py
# Step: 2

#!/usr/bin/env python3
"""
Tower of Hanoi solver for 4 disks
Initial: Peg 0: [4, 3, 2, 1], Peg 1: [], Peg 2: []
Goal: Peg 0: [], Peg 1: [], Peg 2: [4, 3, 2, 1]
"""

def tower_of_hanoi(n, source, target, auxiliary, moves, state):
    """
    Recursive Tower of Hanoi solver
    
    Args:
        n: Number of disks
        source: Source peg (0, 1, or 2)
        target: Target peg (0, 1, or 2)
        auxiliary: Auxiliary peg (0, 1, or 2)
        moves: List to store moves
        state: Current state of pegs
    """
    if n > 0:
        # Move n-1 disks from source to auxiliary
        tower_of_hanoi(n-1, source, auxiliary, target, moves, state)
        
        # Move the nth disk from source to target
        disk = state[source].pop()
        state[target].append(disk)
        moves.append([disk, source, target])
        
        # Move n-1 disks from auxiliary to target
        tower_of_hanoi(n-1, auxiliary, target, source, moves, state)

def print_state(state, step):
    """Print the current state of the pegs"""
    print(f"Step {step}:")
    for i, peg in enumerate(state):
        print(f"  Peg {i}: {peg}")
    print()

def main():
    # Initial state: 4 disks on peg 0
    initial_state = [[4, 3, 2, 1], [], []]
    state = [peg.copy() for peg in initial_state]
    
    print("Initial state:")
    print_state(state, 0)
    
    # Solve Tower of Hanoi
    moves = []
    tower_of_hanoi(4, 0, 2, 1, moves, state)
    
    # Print all moves
    print(f"Optimal solution requires {len(moves)} moves:")
    print("moves = [")
    for i, move in enumerate(moves, 1):
        disk, from_peg, to_peg = move
        print(f"    [{disk}, {from_peg}, {to_peg}],  # Move {i}: Disk {disk} from peg {from_peg} to peg {to_peg}")
    print("]")
    
    # Verify final state
    print("\nFinal state:")
    print_state(state, len(moves))
    
    # Verify the solution
    expected_final = [[], [], [4, 3, 2, 1]]
    if state == expected_final:
        print("✓ Solution verified successfully!")
    else:
        print("✗ Solution verification failed!")
        print(f"Expected: {expected_final}")
        print(f"Got:      {state}")

if __name__ == "__main__":
    main()