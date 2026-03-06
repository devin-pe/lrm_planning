# Tool: execute_python
# Step: 2

def hanoi(n, source, auxiliary, target):
    if n == 1:
        return [[1, source, target]]
    else:
        moves = []
        # Move top n-1 disks from source to auxiliary
        moves.extend(hanoi(n-1, source, target, auxiliary))
        # Move largest disk from source to target
        moves.append([n, source, target])
        # Move n-1 disks from auxiliary to target
        moves.extend(hanoi(n-1, auxiliary, source, target))
        return moves

# Get moves for 5 disks from peg 0 to peg 2 using peg 1 as auxiliary
moves = hanoi(5, 0, 1, 2)
print(moves)