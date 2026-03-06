# Tool: write_file
# Target path: /home/user/hanoi_solver.py
# Step: 2

def tower_of_hanoi(n, source, target, auxiliary):
    """
    Solve Tower of Hanoi puzzle recursively.
    
    Args:
        n: Number of disks
        source: Source peg (0, 1, or 2)
        target: Target peg (0, 1, or 2)
        auxiliary: Auxiliary peg (0, 1, or 2)
    
    Returns:
        List of moves, where each move is [disk_id, from_peg, to_peg]
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

# Solve for 4 disks from peg 0 to peg 2 using peg 1 as auxiliary
moves = tower_of_hanoi(4, 0, 2, 1)

print(f"Total number of moves: {len(moves)}")
print("\nSequence of moves:")
print("moves = [")
for i, move in enumerate(moves, 1):
    print(f"    {move},  # Move {i}: Disk {move[0]} from peg {move[1]} to peg {move[2]}")
print("]")

# Verify the solution by simulating the moves
print("\n" + "="*60)
print("Verifying the solution by simulation:")
print("="*60)

# Initialize pegs
pegs = {
    0: [4, 3, 2, 1],  # Bottom to top
    1: [],
    2: []
}

print("\nInitial state:")
for peg, disks in pegs.items():
    print(f"Peg {peg}: {disks}")

print("\nExecuting moves:")
for i, move in enumerate(moves, 1):
    disk_id, from_peg, to_peg = move
    
    # Validate move
    if not pegs[from_peg]:
        raise ValueError(f"Move {i}: Peg {from_peg} is empty!")
    
    top_disk = pegs[from_peg][-1]
    if top_disk != disk_id:
        raise ValueError(f"Move {i}: Top disk on peg {from_peg} is {top_disk}, not {disk_id}!")
    
    if pegs[to_peg] and pegs[to_peg][-1] < disk_id:
        raise ValueError(f"Move {i}: Cannot place disk {disk_id} on top of smaller disk {pegs[to_peg][-1]}!")
    
    # Execute move
    pegs[from_peg].pop()
    pegs[to_peg].append(disk_id)
    
    print(f"Move {i:2d}: Disk {disk_id} from peg {from_peg} to peg {to_peg}")
    print(f"         State: Peg 0: {pegs[0]}, Peg 1: {pegs[1]}, Peg 2: {pegs[2]}")

print("\n" + "="*60)
print("Final state:")
for peg, disks in pegs.items():
    print(f"Peg {peg}: {disks}")

# Check if goal is reached
if pegs[0] == [] and pegs[1] == [] and pegs[2] == [4, 3, 2, 1]:
    print("\n✓ SUCCESS: Goal configuration reached!")
else:
    print("\n✗ FAILURE: Goal configuration not reached!")