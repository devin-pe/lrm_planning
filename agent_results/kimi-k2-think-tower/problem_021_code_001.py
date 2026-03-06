# Tool: write_file
# Target path: /home/user/hanoi_solver.py
# Step: 2

def tower_of_hanoi(n, source, target, auxiliary):
    """
    Solve Tower of Hanoi puzzle and return list of moves.
    
    Args:
        n: Number of disks
        source: Source peg (0, 1, or 2)
        target: Target peg (0, 1, or 2)
        auxiliary: Auxiliary peg (0, 1, or 2)
    
    Returns:
        List of moves in format [[disk_id, from_peg, to_peg], ...]
    """
    moves = []
    
    def hanoi(n, source, target, auxiliary):
        if n == 1:
            # Base case: move a single disk
            moves.append([n, source, target])
        else:
            # Move n-1 disks from source to auxiliary
            hanoi(n-1, source, auxiliary, target)
            # Move the nth disk from source to target
            moves.append([n, source, target])
            # Move n-1 disks from auxiliary to target
            hanoi(n-1, auxiliary, target, source)
    
    hanoi(n, source, target, auxiliary)
    return moves

# Solve for 5 disks
print("Solving Tower of Hanoi with 5 disks...")
moves = tower_of_hanoi(5, 0, 2, 1)

print(f"Total moves: {len(moves)}")
print("\nSequence of moves:")
for i, move in enumerate(moves, 1):
    disk, from_peg, to_peg = move
    print(f"{i:2d}. Move disk {disk} from peg {from_peg} to peg {to_peg}")

print("\nComplete list of moves:")
print(f"moves = {moves}")