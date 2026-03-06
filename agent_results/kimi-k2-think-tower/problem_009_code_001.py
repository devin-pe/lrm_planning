# Tool: execute_python
# Step: 2


def tower_of_hanoi(n, source, target, auxiliary):
    """
    Solve Tower of Hanoi puzzle and return list of moves.
    
    Args:
        n: Number of disks
        source: Source peg (0-indexed)
        target: Target peg (0-indexed)
        auxiliary: Auxiliary peg (0-indexed)
    
    Returns:
        List of moves in format [disk_id, from_peg, to_peg]
    """
    moves = []
    
    def move_disks(num_disks, src, tgt, aux):
        if num_disks == 1:
            # Base case: move single disk
            moves.append([1, src, tgt])
        else:
            # Recursive case:
            # 1. Move n-1 disks from source to auxiliary
            move_disks(num_disks - 1, src, aux, tgt)
            # 2. Move disk n from source to target
            moves.append([num_disks, src, tgt])
            # 3. Move n-1 disks from auxiliary to target
            move_disks(num_disks - 1, aux, tgt, src)
    
    move_disks(n, source, target, auxiliary)
    return moves

# Solve for 3 disks
moves = tower_of_hanoi(3, 0, 2, 1)
print("Solution moves:")
for i, move in enumerate(moves, 1):
    print(f"{i}. Move disk {move[0]} from peg {move[1]} to peg {move[2]}")

print("\nComplete moves list:")
print(moves)
