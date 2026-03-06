# Tool: write_file
# Target path: /home/user/hanoi_solver.py
# Step: 2

def solve_hanoi(n, source, target, auxiliary):
    """
    Solve Tower of Hanoi puzzle and return list of moves.
    Each move is [disk_id, from_peg, to_peg]
    """
    moves = []
    
    def hanoi(n, source, target, auxiliary):
        if n == 1:
            # Move disk 1 from source to target
            moves.append([1, source, target])
        else:
            # Move n-1 disks from source to auxiliary
            hanoi(n-1, source, auxiliary, target)
            # Move disk n from source to target
            moves.append([n, source, target])
            # Move n-1 disks from auxiliary to target
            hanoi(n-1, auxiliary, target, source)
    
    hanoi(n, source, target, auxiliary)
    return moves

# Solve for 3 disks from peg 0 to peg 2 using peg 1 as auxiliary
moves = solve_hanoi(3, 0, 2, 1)

print("Optimal sequence of moves:")
print(f"moves = {moves}")
print(f"\nTotal moves: {len(moves)}")

# Verify the solution by simulating the moves
print("\nVerifying the solution:")
pegs = [[3, 2, 1], [], []]  # Initial state
print(f"Initial state: {pegs}")

for i, move in enumerate(moves):
    disk_id, from_peg, to_peg = move
    
    # Check if move is valid
    if not pegs[from_peg]:
        print(f"ERROR: Move {i+1} - Peg {from_peg} is empty!")
        break
    
    top_disk = pegs[from_peg][-1]
    if top_disk != disk_id:
        print(f"ERROR: Move {i+1} - Expected disk {disk_id} but found disk {top_disk} on peg {from_peg}!")
        break
    
    # Check if target peg is empty or has larger disk
    if pegs[to_peg] and pegs[to_peg][-1] < disk_id:
        print(f"ERROR: Move {i+1} - Cannot place disk {disk_id} on top of disk {pegs[to_peg][-1]}!")
        break
    
    # Make the move
    pegs[from_peg].pop()
    pegs[to_peg].append(disk_id)
    print(f"Move {i+1}: Disk {disk_id} from peg {from_peg} to peg {to_peg} -> {pegs}")

# Check final state
if pegs == [[], [], [3, 2, 1]]:
    print("\n✓ SUCCESS! Final state matches goal configuration.")
else:
    print(f"\n✗ FAILURE! Final state {pegs} does not match goal [[], [], [3, 2, 1]]")