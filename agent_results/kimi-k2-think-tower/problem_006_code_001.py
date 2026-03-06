# Tool: write_file
# Target path: /home/user/hanoi_solver.py
# Step: 2

#!/usr/bin/env python3
"""
Tower of Hanoi solver for 3 disks.
Verifies the optimal sequence of moves from initial to goal configuration.
"""

def apply_move(state, move):
    """Apply a move to the current state and return new state."""
    disk, from_peg, to_peg = move
    
    # Create a deep copy of the state
    new_state = [peg.copy() for peg in state]
    
    # Remove disk from source peg
    if not new_state[from_peg] or new_state[from_peg][-1] != disk:
        raise ValueError(f"Invalid move: Disk {disk} not on top of peg {from_peg}")
    
    new_state[from_peg].pop()
    
    # Add disk to destination peg
    if new_state[to_peg] and new_state[to_peg][-1] < disk:
        raise ValueError(f"Invalid move: Cannot place disk {disk} on smaller disk {new_state[to_peg][-1]}")
    
    new_state[to_peg].append(disk)
    
    return new_state

def print_state(state, step):
    """Print the current state of the pegs."""
    print(f"Step {step}: Peg 0: {state[0]}, Peg 1: {state[1]}, Peg 2: {state[2]}")

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
    
    current_state = initial_state
    
    for i, move in enumerate(moves, 1):
        disk, from_peg, to_peg = move
        print(f"\nMove {i}: Move disk {disk} from peg {from_peg} to peg {to_peg}")
        
        try:
            current_state = apply_move(current_state, move)
            print_state(current_state, i)
        except ValueError as e:
            print(f"ERROR: {e}")
            return False
    
    print("\n" + "=" * 40)
    print("Final state:")
    print_state("Final", current_state)
    
    # Verify if we reached the goal
    if current_state == goal_state:
        print("\n✓ SUCCESS! Goal configuration reached.")
        print("\nOptimal sequence of moves:")
        print(f"moves = {moves}")
        return True
    else:
        print(f"\n✗ FAILED! Final state does not match goal.")
        print(f"Expected: {goal_state}")
        print(f"Got:      {current_state}")
        return False

if __name__ == "__main__":
    solve_hanoi_3_disks()