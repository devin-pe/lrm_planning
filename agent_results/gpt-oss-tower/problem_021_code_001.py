# Tool: execute_python
# Step: 2

def hanoi(n, source, target, auxiliary, moves):
    if n == 0:
        return
    # move n-1 disks from source to auxiliary
    hanoi(n-1, source, auxiliary, target, moves)
    # move nth disk from source to target
    moves.append([n, source, target])
    # move n-1 disks from auxiliary to target
    hanoi(n-1, auxiliary, target, source, moves)

moves = []
hanoi(5, 0, 2, 1, moves)
print('Number of moves:', len(moves))
print(moves)
