# Tool: execute_python
# Step: 2

def hanoi(n, source, auxiliary, target):
    if n == 1:
        return [[1, source, target]]
    else:
        # Move n-1 disks from source to auxiliary peg
        moves = hanoi(n-1, source, target, auxiliary)
        # Move the nth disk from source to target
        moves.append([n, source, target])
        # Move n-1 disks from auxiliary to target
        moves += hanoi(n-1, auxiliary, source, target)
        return moves

# Generate moves for 3 disks from peg0 to peg2 using peg1 as auxiliary
moves_sequence = hanoi(3, 0, 1, 2)
print(f'moves = {moves_sequence}')