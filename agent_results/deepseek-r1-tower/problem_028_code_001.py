# Tool: execute_python
# Step: 2

def hanoi(n, source, target, auxiliary):
    if n == 1:
        return [[1, source, target]]
    else:
        moves = []
        # Move n-1 disks from source to auxiliary
        moves += hanoi(n-1, source, auxiliary, target)
        # Move the nth disk from source to target
        moves.append([n, source, target])
        # Move n-1 disks from auxiliary to target
        moves += hanoi(n-1, auxiliary, target, source)
        return moves

# Generate moves for 5 disks from peg 0 to peg 2 using peg 1 as auxiliary
moves = hanoi(5, 0, 2, 1)
print(f'Total moves: {len(moves)}')
print(moves)