# Tool: execute_python
# Step: 2

def tower_of_hanoi(n, source, auxiliary, target, moves=None):
    if moves is None:
        moves = []
    if n == 1:
        moves.append([1, source, target])
    else:
        tower_of_hanoi(n-1, source, target, auxiliary, moves)
        moves.append([n, source, target])
        tower_of_hanoi(n-1, auxiliary, source, target, moves)
    return moves

moves = tower_of_hanoi(5, 0, 1, 2)
print(moves)