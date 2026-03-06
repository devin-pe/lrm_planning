# Tool: execute_python
# Step: 4

def simulate(moves, n=5):
    pegs = [list(range(n,0,-1)), [], []]
    for move in moves:
        disk, src, dst = move
        # check top of src
        if not pegs[src] or pegs[src][-1] != disk:
            raise Exception(f'Invalid move {move}: top of src is {pegs[src][-1] if pegs[src] else None}')
        # check dst
        if pegs[dst] and pegs[dst][-1] < disk:
            raise Exception(f'Invalid move {move}: cannot place larger on smaller')
        pegs[src].pop()
        pegs[dst].append(disk)
    return pegs

moves = [[1, 0, 2], [2, 0, 1], [1, 2, 1], [3, 0, 2], [1, 1, 0], [2, 1, 2], [1, 0, 2], [4, 0, 1], [1, 2, 1], [2, 2, 0], [1, 1, 0], [3, 2, 1], [1, 0, 2], [2, 0, 1], [1, 2, 1], [5, 0, 2], [1, 1, 0], [2, 1, 2], [1, 0, 2], [3, 1, 0], [1, 2, 1], [2, 2, 0], [1, 1, 0], [4, 1, 2], [1, 0, 2], [2, 0, 1], [1, 2, 1], [3, 0, 2], [1, 1, 0], [2, 1, 2], [1, 0, 2]]
print(simulate(moves))
