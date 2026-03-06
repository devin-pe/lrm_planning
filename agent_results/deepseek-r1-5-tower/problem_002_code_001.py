# Tool: execute_python
# Step: 2

def hanoi(n, source, auxiliary, target, moves):
    if n == 1:
        moves.append([1, source, target])
    else:
        hanoi(n-1, source, target, auxiliary, moves)
        moves.append([n, source, target])
        hanoi(n-1, auxiliary, source, target, moves)

moves = []
hanoi(4, 0, 1, 2, moves)
print("moves =", moves)