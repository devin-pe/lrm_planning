# Tool: write_file
# Target path: /home/user/hanoi_solver.py
# Step: 2

#!/usr/bin/env python3
"""
Tower of Hanoi Solver for 3 disks
Initial: Peg 0 has [3, 2, 1], Pegs 1 and 2 are empty
Goal: Peg 2 has [3, 2, 1], Pegs 0 and 1 are empty
"""

def tower_of_hanoi(n, source, target, auxiliary):
    """
    Solve Tower of Hanoi puzzle recursively.
    Returns list of moves in format [disk_id, from_peg, to_peg]
    """
    moves = []
    
    if n == 1:
        # Base case: move single disk
        moves.append([1, source, target])
        return moves
    
    # Move n-1 disks from source to auxiliary
    moves.extend(tower_of_hanoi(n-1, source, auxiliary, target))
    
    # Move nth disk from source to target
    moves.append([n, source, target])
    
    # Move n-1 disks from auxiliary to target
    moves.extend(tower_of_hanoi(n-1, auxiliary, target, source))
    
    return moves

# Solve for 3 disks from peg 0 to peg 2 using peg 1 as auxiliary
moves = tower_of_hanoi(3, 0, 2, 1)

print("Tower of Hanoi Solution for 3 disks:")
print("=" * 50)
print(f"Total moves: {len(moves)}")
print()
print("Sequence of moves:")
for i, move in enumerate(moves, 1):
    disk_id, from_peg, to_peg = move
    print(f"{i}. Move disk {disk_id} from peg {from_peg} to peg {to_peg}")

print()
print("Complete list of moves:")
print(f"moves = {moves}")

# Verify the solution step by step
print()
print("Verification:")
print("=" * 50)

# Initialize pegs
pegs = {0: [3, 2, 1], 1: [], 2: []}
print(f"Initial state: {pegs}")

for i, move in enumerate(moves, 1):
    disk_id, from_peg, to_peg = move
    
    # Check if move is valid
    if not pegs[from_peg]:
        print(f"ERROR at step {i}: Peg {from_peg} is empty!")
        break
    
    top_disk = pegs[from_peg][-1]
    if top_disk != disk_id:
        print(f"ERROR at step {i}: Top disk on peg {from_peg} is {top_disk}, not {disk_id}!")
        break
    
    # Check if target peg is empty or has larger disk
    if pegs[to_peg] and pegs[to_peg][-1] < disk_id:
        print(f"ERROR at step {i}: Cannot place disk {disk_id} on top of disk {pegs[to_peg][-1]}!")
        break
    
    # Make the move
    pegs[from_peg].pop()
    pegs[to_peg].append(disk_id)
    
    print(f"Step {i}: Move disk {disk_id} from peg {from_peg} to peg {to_peg}")
    print(f"  State: {pegs}")

# Check final state
if pegs[0] == [] and pegs[1] == [] and pegs[2] == [3, 2, 1]:
    print()
    print("✓ SUCCESS! Final state matches goal configuration.")
else:
    print()
    print("✗ FAILURE! Final state does not match goal configuration.")
    print(f"  Expected: {{0: [], 1: [], 2: [3, 2, 1]}}")
    print(f"  Got: {pegs}")