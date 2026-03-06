# Tool: execute_python
# Step: 2

def hanoi(n, source, auxiliary, target):
    if n == 1:
        return [[1, source, target]]
    else:
        moves = hanoi(n-1, source, target, auxiliary)
        moves.append([n, source, target])
        moves.extend(hanoi(n-1, auxiliary, source, target))
        return moves

moves = hanoi(5, 0, 1, 2)
print(moves)