# Tool: write_file
# Target path: /home/user/hanoi_solver.py
# Step: 2

def tower_of_hanoi(n, source, target, auxiliary):
    """
    Solve Tower of Hanoi puzzle recursively.
    Returns list of moves in format [disk_id, from_peg, to_peg]
    """
    moves = []
    
    def hanoi(n, source, target, auxiliary):
        if n > 0:
            # Move n-1 disks from source to auxiliary
            hanoi(n-1, source, auxiliary, target)
            
            # Move the nth disk from source to target
            moves.append([n, source, target])
            
            # Move n-1 disks from auxiliary to target
            hanoi(n-1, auxiliary, target, source)
    
    hanoi(n, source, target, auxiliary)
    return moves

# Solve for 4 disks from peg 0 to peg 2
moves = tower_of_hanoi(4, 0, 2, 1)

print(f"Number of moves: {len(moves)}")
print("\nSequence of moves:")
print("moves = [")
for i, move in enumerate(moves, 1):
    disk, from_peg, to_peg = move
    print(f"    {move},  # Move {i}: Disk {disk} from peg {from_peg} to peg {to_peg}")
print("]")

# Verify the solution
print("\n" + "="*50)
print("VERIFICATION:")
print("="*50)

# Initialize pegs
pegs = {0: [4, 3, 2, 1], 1: [], 2: []}
print(f"\nInitial state: {pegs}")

for i, move in enumerate(moves, 1):
    disk, from_peg, to_peg = move
    
    # Check if move is valid
    if not pegs[from_peg] or pegs[from_peg][-1] != disk:
        print(f"ERROR at move {i}: Disk {disk} is not on top of peg {from_peg}")
        break
    
    if pegs[to_peg] and pegs[to_peg][-1] < disk:
        print(f"ERROR at move {i}: Cannot place disk {disk} on top of smaller disk {pegs[to_peg][-1]}")
        break
    
    # Execute move
    pegs[from_peg].pop()
    pegs[to_peg].append(disk)
    
    print(f"After move {i} ({disk}: {from_peg}→{to_peg}): {pegs}")

# Check final state
if pegs[0] == [] and pegs[1] == [] and pegs[2] == [4, 3, 2, 1]:
    print("\n✓ SUCCESS! Final configuration matches goal state.")
else:
    print(f"\n✗ FAILURE! Final state: {pegs}")