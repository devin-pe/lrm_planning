# Tool: execute_python
# Step: 2

def tower_of_hanoi(n, from_peg, to_peg, aux_peg, moves):
    if n == 1:
        moves.append([1, from_peg, to_peg])
        return
    tower_of_hanoi(n-1, from_peg, aux_peg, to_peg, moves)
    moves.append([n, from_peg, to_peg])
    tower_of_hanoi(n-1, aux_peg, to_peg, from_peg, moves)

def solve_tower_of_hanoi(n):
    moves = []
    tower_of_hanoi(n, 0, 2, 1, moves)
    return moves

# Test the function
n = 3
moves = solve_tower_of_hanoi(n)
print("moves =", moves)