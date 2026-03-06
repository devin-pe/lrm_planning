# Tool: execute_python
# Step: 2

moves = []

def hanoi(n, source, auxiliary, target):
    if n == 1:
        moves.append([1, source, target])
    else:
        hanoi(n-1, source, target, auxiliary)
        moves.append([n, source, target])
        hanoi(n-1, auxiliary, source, target)

hanoi(3, 0, 1, 2)
print(moves)