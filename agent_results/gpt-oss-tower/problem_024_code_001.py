# Tool: execute_python
# Step: 2

def hanoi(n, source, target, auxiliary, moves=None):
    if moves is None:
        moves = []
    if n == 1:
        moves.append([1, source, target])
    else:
        hanoi(n-1, source, auxiliary, target, moves)
        moves.append([n, source, target])
        hanoi(n-1, auxiliary, target, source, moves)
    return moves

moves = hanoi(5, 0, 2, 1)
print(len(moves))
print(moves)
