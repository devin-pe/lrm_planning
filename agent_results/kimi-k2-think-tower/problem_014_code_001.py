# Tool: write_file
# Target path: /home/user/hanoi_solver.py
# Step: 2

def tower_of_hanoi(n, source, target, auxiliary, moves=None):
    """Generate moves for Tower of Hanoi puzzle.
    
    Args:
        n: Number of disks
        source: Source peg (0-indexed)
        target: Target peg (0-indexed)
        auxiliary: Auxiliary peg (0-indexed)
        moves: List to store moves
    
    Returns:
        List of moves in format [[disk, from_peg, to_peg], ...]
    """
    if moves is None:
        moves = []
    
    if n == 1:
        # Base case: move single disk
        moves.append([1, source, target])
    else:
        # Recursive case:
        # 1. Move n-1 disks from source to auxiliary
        tower_of_hanoi(n-1, source, auxiliary, target, moves)
        
        # 2. Move nth disk from source to target
        moves.append([n, source, target])
        
        # 3. Move n-1 disks from auxiliary to target
        tower_of_hanoi(n-1, auxiliary, target, source, moves)
    
    return moves

# Solve for 4 disks from peg 0 to peg 2 using peg 1 as auxiliary
moves = tower_of_hanoi(4, 0, 2, 1)

print(f"Number of moves: {len(moves)}")
print("\nSequence of moves:")
for i, move in enumerate(moves, 1):
    disk, from_peg, to_peg = move
    print(f"{i}. Move disk {disk} from peg {from_peg} to peg {to_peg}")

print(f"\nmoves = {moves}")

# Verify the solution by simulating the moves
print("\n" + "="*50)
print("Verifying the solution...")
print("="*50)

# Initialize pegs
pegs = {
    0: [4, 3, 2, 1],  # bottom to top
    1: [],
    2: []
}

print(f"\nInitial state:")
for peg in range(3):
    print(f"Peg {peg}: {pegs[peg]}")

# Simulate each move
for i, move in enumerate(moves, 1):
    disk, from_peg, to_peg = move
    
    # Check if move is valid
    if not pegs[from_peg]:
        raise ValueError(f"Move {i}: Peg {from_peg} is empty!")
    
    top_disk = pegs[from_peg][-1]
    if top_disk != disk:
        raise ValueError(f"Move {i}: Top disk on peg {from_peg} is {top_disk}, not {disk}!")
    
    if pegs[to_peg] and pegs[to_peg][-1] < disk:
        raise ValueError(f"Move {i}: Cannot place disk {disk} on top of smaller disk {pegs[to_peg][-1]}!")
    
    # Make the move
    pegs[from_peg].pop()
    pegs[to_peg].append(disk)
    
    print(f"\nAfter move {i} (disk {disk} from {from_peg} to {to_peg}):")
    for peg in range(3):
        print(f"Peg {peg}: {pegs[peg]}")

# Check final state
if pegs[0] == [] and pegs[1] == [] and pegs[2] == [4, 3, 2, 1]:
    print("\n✓ Solution verified successfully!")
else:
    print(f"\n✗ Solution failed! Final state: {pegs}")