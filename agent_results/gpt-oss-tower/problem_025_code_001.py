# Tool: execute_python
# Step: 2

def hanoi(n, source, target, auxiliary, moves):
    if n == 0:
        return
    hanoi(n-1, source, auxiliary, target, moves)
    moves.append([n, source, target])
    hanoi(n-1, auxiliary, target, source, moves)

moves = []
hanoi(5, 0, 2, 1, moves)
print(moves)
