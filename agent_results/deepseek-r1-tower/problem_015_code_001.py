# Tool: execute_python
# Step: 2

def hanoi(n, source, auxiliary, destination, moves):
    if n == 1:
        moves.append([1, source, destination])
    else:
        hanoi(n-1, source, destination, auxiliary, moves)
        moves.append([n, source, destination])
        hanoi(n-1, auxiliary, source, destination, moves)

moves = []
hanoi(4, 0, 1, 2, moves)
print("moves =", moves)