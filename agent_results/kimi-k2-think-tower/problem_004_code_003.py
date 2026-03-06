# Tool: write_file
# Target path: /home/user/hanoi_solver.py
# Step: 8

#!/usr/bin/env python3
"""
Tower of Hanoi solver for 3 disks.
Verifies the sequence of moves from initial to goal configuration.
"""

def apply_move(state, move):
    """Apply a single move to the state."""
    disk, from_peg, to_peg = move
    
    # Check that the move is valid
    if not state[from_peg]:
        raise ValueError(f"Peg {from_peg} is empty, cannot move disk {disk}")
    
    if state[from_peg][-1] != disk:
        raise ValueError(f"Disk {disk} is not on top of peg {from_peg}")
    
    if state[to_peg] and state[to_peg][-1] < disk:
        raise ValueError(f"Cannot place disk {disk} on top of smaller disk {state[to_peg][-1]}")
    
    # Apply the move
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
    # Initial state: Peg 0 has [3, 2, 1], others are empty
    initial_state = [[3, 2, 1], [], []]
    
    # Goal state: Peg 2 has [3, 2, 1], others are empty
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
    
    print("Tower of Hanoi - 3 Disks Solution")
    print("=" * 40)
    print("\nInitial state:")
    print_state(initial_state, 0)
    
    state = [peg.copy() for peg in initial_state]
    
    for i, move in enumerate(moves, 1):
        disk, from_peg, to_peg = move
        print(f"Move {i}: Move disk {disk} from peg {from_peg} to peg {to_peg}")
        
        state = apply_move(state, move)  # Fixed: pass move instead of i
        print_state(state, i)
    
    print("Goal reached!" if state == goal_state else "Goal not reached!")
    
    return moves, state == goal_state

if __name__ == "__main__":
    moves, success = solve_hanoi_3_disks()
    
    print("\n" + "=" * 40)
    print("FINAL ANSWER:")
    print("=" * 40)
    print(f"moves = {moves}")
    print(f"\nVerification: {'SUCCESS' if success else 'FAILED'}")