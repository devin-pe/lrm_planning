# Tool: execute_python
# Step: 2

def hanoi(n, source, target, auxiliary):
    if n == 1:
        return [[1, source, target]]
    else:
        moves = []
        moves.extend(hanoi(n-1, source, auxiliary, target))
        moves.append([n, source, target])
        moves.extend(hanoi(n-1, auxiliary, target, source))
        return moves

moves = hanoi(3, 0, 2, 1)
print(moves)