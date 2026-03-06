# Tool: write_file
# Target path: /home/user/hanoi_solver.py
# Step: 2

def tower_of_hanoi(n, source, target, auxiliary, moves=None):
    """
    Solve Tower of Hanoi puzzle recursively.
    
    Args:
        n: Number of disks
        source: Source peg
        target: Target peg
        auxiliary: Auxiliary peg
        moves: List to store moves
    
    Returns:
        List of moves in format [disk_id, from_peg, to_peg]
    """
    if moves is None:
        moves = []
    
    if n == 1:
        # Base case: move a single disk
        moves.append([1, source, target])
        return moves
    
    # Move n-1 disks from source to auxiliary
    tower_of_hanoi(n-1, source, auxiliary, target, moves)
    
    # Move the nth disk from source to target
    moves.append([n, source, target])
    
    # Move n-1 disks from auxiliary to target
    tower_of_hanoi(n-1, auxiliary, target, source, moves)
    
    return moves

# Solve for 4 disks from peg 0 to peg 2 using peg 1 as auxiliary
moves = tower_of_hanoi(4, 0, 2, 1)

print(f"Total moves: {len(moves)}")
print("\nSequence of moves:")
for i, move in enumerate(moves, 1):
    disk, from_peg, to_peg = move
    print(f"{i}. Move disk {disk} from peg {from_peg} to peg {to_peg}")

print("\nComplete list of moves:")
print(f"moves = {moves}")

# Verify the solution
print("\n" + "="*50)
print("VERIFYING THE SOLUTION:")
print("="*50)

# Initialize pegs
pegs = {0: [4, 3, 2, 1], 1: [], 2: []}
print(f"\nInitial state: {pegs}")

for i, move in enumerate(moves, 1):
    disk, from_peg, to_peg = move
    
    # Check if the move is valid
    if not pegs[from_peg]:
        print(f"ERROR at move {i}: Peg {from_peg} is empty!")
        break
    
    top_disk = pegs[from_peg][-1]
    if top_disk != disk:
        print(f"ERROR at move {i}: Top disk on peg {from_peg} is {top_disk}, not {disk}!")
        break
    
    # Check if target peg has a smaller disk
    if pegs[to_peg] and pegs[to_peg][-1] < disk:
        print(f"ERROR at move {i}: Cannot place disk {disk} on top of disk {pegs[to_peg][-1]}!")
        break
    
    # Make the move
    pegs[from_peg].pop()
    pegs[to_peg].append(disk)
    
    print(f"After move {i} ({disk}, {from_peg}, {to_peg}): {pegs}")

# Check final state
if pegs[0] == [] and pegs[1] == [] and pegs[2] == [4, 3, 2, 1]:
    print("\n✓ SUCCESS! Final state matches goal configuration.")
else:
    print(f"\n✗ FAILURE! Final state {pegs} does not match goal.")