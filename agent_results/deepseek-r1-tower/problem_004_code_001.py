# Tool: execute_python
# Step: 2

def hanoi_moves(n, source, auxiliary, destination):
    if n == 1:
        return [[1, source, destination]]
    else:
        moves = []
        moves += hanoi_moves(n-1, source, destination, auxiliary)
        moves.append([n, source, destination])
        moves += hanoi_moves(n-1, auxiliary, source, destination)
        return moves

moves = hanoi_moves(3, 0, 1, 2)
for move in moves:
    print(move)