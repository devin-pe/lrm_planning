# Tool: execute_python
# Step: 2

# Initialize the pegs
pegs = [[], [], []]
# Initial configuration: Disk 3 (largest) at bottom, Disk 2 in middle, Disk 1 (smallest) on top of peg 0
pegs[0] = [3, 2, 1]
pegs[1] = []
pegs[2] = []

print("Initial configuration:")
print(f"Peg 0: {pegs[0]}")
print(f"Peg 1: {pegs[1]}") 
print(f"Peg 2: {pegs[2]}")
print()

# Define the moves according to the Tower of Hanoi solution
moves = [
    [1, 0, 2],   # Move disk 1 from peg 0 to peg 2
    [2, 0, 1],   # Move disk 2 from peg 0 to peg 1
    [1, 2, 1],   # Move disk 1 from peg 2 to peg 1
    [3, 0, 2],   # Move disk 3 from peg 0 to peg 2
    [1, 1, 0],   # Move disk 1 from peg 1 to peg 0
    [2, 1, 2],   # Move disk 2 from peg 1 to peg 2
    [1, 0, 2]    # Move disk 1 from peg 0 to peg 2
]

# Apply each move
for i, move in enumerate(moves, 1):
    disk, from_peg, to_peg = move
    
    # Check that the move is valid
    if not pegs[from_peg] or pegs[from_peg][-1] != disk:
        print(f"ERROR in move {i}: Cannot move disk {disk} from peg {from_peg}. Top disk is {pegs[from_peg][-1] if pegs[from_peg] else 'empty'}")
        break
    
    if pegs[to_peg] and pegs[to_peg][-1] < disk:
        print(f"ERROR in move {i}: Cannot place larger disk {disk} on smaller disk {pegs[to_peg][-1]}")
        break
    
    # Perform the move
    pegs[from_peg].pop()
    pegs[to_peg].append(disk)
    
    print(f"Move {i}: Move disk {disk} from peg {from_peg} to peg {to_peg}")
    print(f"  Peg 0: {pegs[0]}")
    print(f"  Peg 1: {pegs[1]}")
    print(f"  Peg 2: {pegs[2]}")
    print()

# Verify final configuration
print("Final configuration:")
print(f"Peg 0: {pegs[0]}")
print(f"Peg 1: {pegs[1]}")
print(f"Peg 2: {pegs[2]}")
print()

# Check if we reached the goal
if pegs[0] == [] and pegs[1] == [] and pegs[2] == [3, 2, 1]:
    print("✓ SUCCESS! Goal configuration reached!")
else:
    print("✗ FAILED! Goal configuration not reached.")