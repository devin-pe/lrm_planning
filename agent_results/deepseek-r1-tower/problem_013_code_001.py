# Tool: execute_python
# Step: 2

def hanoi(n, source, target, auxiliary, moves):
    if n > 0:
        # Move n-1 disks from source to auxiliary
        hanoi(n-1, source, auxiliary, target, moves)
        
        # Move the nth disk from source to target
        disk_id = n
        moves.append([disk_id, source, target])
        
        # Move n-1 disks from auxiliary to target
        hanoi(n-1, auxiliary, target, source, moves)

# Initialize moves list
moves = []

# Solve for 4 disks from peg 0 to peg 2 using peg 1 as auxiliary
hanoi(4, 0, 2, 1, moves)

# Return the moves
moves