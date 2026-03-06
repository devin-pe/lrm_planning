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
    disk_id, from_peg, to_peg = move
    
    # Check that the move is valid
    if not state[from_peg]:
        raise ValueError(f"Cannot move from empty peg {from_peg}")
    
    top_disk = state[from_peg][-1]
    if top_disk != disk_id:
        raise ValueError(f"Disk {disk_id} is not on top of peg {from_peg}")
    
    if state[to_peg] and state[to_peg][-1] < disk_id:
        raise ValueError(f"Cannot place larger disk {disk_id} on smaller disk {state[to_peg][-1]}")
    
    # Apply the move
    state[from_peg].pop()
    state[to_peg].append(disk_id)
    
    return state

def print_state(state, move_num=None):
    """Print the current state."""
    if move_num is not None:
        print(f"\nAfter move {move_num}:")
    for i, peg in enumerate(state):
        print(f"Peg {i}: {peg}")

def solve_hanoi_3disks():
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
    
    print("Initial state:")
    print_state(initial_state)
    
    state = [peg.copy() for peg in initial_state]
    
    for i, move in enumerate(moves, 1):
        disk_id, from_peg, to_peg = move
        print(f"\nMove {i}: Disk {disk_id} from peg {from_peg} to peg {to_peg}")
        state = apply_move(state, move)
        print_state(state, i)
    
    print("\n" + "="*50)
    if state == goal_state:
        print("✓ SUCCESS! Reached goal configuration.")
        print(f"✓ Total moves: {len(moves)}")
        print(f"✓ Optimal solution (minimum moves for 3 disks is 2^3 - 1 = 7)")
    else:
        print("✗ FAILED! Did not reach goal configuration.")
        print(f"Expected: {goal_state}")
        print(f"Got:      {state}")
    
    return moves

if __name__ == "__main__":
    moves = solve_hanoi_3disks()
    print("\n" + "="*50)
    print("FINAL ANSWER:")
    print("moves =", moves)