# Tool: write_file
# Target path: /home/user/hanoi_solver.py
# Step: 2

#!/usr/bin/env python3
"""
Tower of Hanoi solver for 4 disks
Simulates the moves to verify correctness
"""

def simulate_hanoi(moves, initial_state):
    """Simulate the Tower of Hanoi moves and verify validity"""
    state = [peg.copy() for peg in initial_state]
    
    print("Initial state:")
    for i, peg in enumerate(state):
        print(f"Peg {i}: {peg}")
    print()
    
    for move_num, (disk, from_peg, to_peg) in enumerate(moves, 1):
        print(f"Move {move_num}: Disk {disk} from peg {from_peg} to peg {to_peg}")
        
        # Validate move
        if not state[from_peg]:
            print(f"ERROR: Peg {from_peg} is empty!")
            return False
        
        top_disk = state[from_peg][-1]
        if top_disk != disk:
            print(f"ERROR: Top disk on peg {from_peg} is {top_disk}, not {disk}!")
            return False
        
        if state[to_peg] and state[to_peg][-1] < disk:
            print(f"ERROR: Cannot place larger disk {disk} on smaller disk {state[to_peg][-1]}!")
            return False
        
        # Execute move
        state[from_peg].pop()
        state[to_peg].append(disk)
        
        print("State after move:")
        for i, peg in enumerate(state):
            print(f"Peg {i}: {peg}")
        print()
    
    print("Final state:")
    for i, peg in enumerate(state):
        print(f"Peg {i}: {peg}")
    
    # Check if goal reached
    if state[0] == [] and state[1] == [] and state[2] == [4, 3, 2, 1]:
        print("\n✓ Goal reached successfully!")
        return True
    else:
        print("\n✗ Goal not reached!")
        return False

# Initial state: Peg 0 has disks 4,3,2,1 (bottom to top)
initial_state = [
    [4, 3, 2, 1],  # Peg 0
    [],             # Peg 1
    []              # Peg 2
]

# Optimal sequence of moves for 4 disks (15 moves)
moves = [
    [1, 0, 1],  # Move disk 1 from peg 0 to peg 1
    [2, 0, 2],  # Move disk 2 from peg 0 to peg 2
    [1, 1, 2],  # Move disk 1 from peg 1 to peg 2
    [3, 0, 1],  # Move disk 3 from peg 0 to peg 1
    [1, 2, 0],  # Move disk 1 from peg 2 to peg 0
    [2, 2, 1],  # Move disk 2 from peg 2 to peg 1
    [1, 0, 1],  # Move disk 1 from peg 0 to peg 1
    [4, 0, 2],  # Move disk 4 from peg 0 to peg 2
    [1, 1, 2],  # Move disk 1 from peg 1 to peg 2
    [2, 1, 0],  # Move disk 2 from peg 1 to peg 0
    [1, 2, 0],  # Move disk 1 from peg 2 to peg 0
    [3, 1, 2],  # Move disk 3 from peg 1 to peg 2
    [1, 0, 1],  # Move disk 1 from peg 0 to peg 1
    [2, 0, 2],  # Move disk 2 from peg 0 to peg 2
    [1, 1, 2],  # Move disk 1 from peg 1 to peg 2
]

print("Tower of Hanoi - 4 disks solution verification")
print("=" * 50)
success = simulate_hanoi(moves, initial_state)

if success:
    print(f"\nTotal moves: {len(moves)}")
    print("\nComplete list of moves:")
    print("moves = [")
    for move in moves:
        print(f"    {move},")
    print("]")
else:
    print("\nInvalid solution!")