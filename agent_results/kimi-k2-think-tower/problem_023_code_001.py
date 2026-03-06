# Tool: execute_python
# Step: 2


def tower_of_hanoi(n, source, target, auxiliary):
    """
    Solve Tower of Hanoi puzzle recursively.
    
    Args:
        n: Number of disks
        source: Source peg (0, 1, or 2)
        target: Target peg (0, 1, or 2)
        auxiliary: Auxiliary peg (0, 1, or 2)
    
    Returns:
        List of moves in format [[disk_id, from_peg, to_peg], ...]
    """
    moves = []
    
    if n == 1:
        # Base case: move single disk from source to target
        moves.append([1, source, target])
        return moves
    
    # Step 1: Move n-1 disks from source to auxiliary (using target as auxiliary)
    moves.extend(tower_of_hanoi(n-1, source, auxiliary, target))
    
    # Step 2: Move nth disk from source to target
    moves.append([n, source, target])
    
    # Step 3: Move n-1 disks from auxiliary to target (using source as auxiliary)
    moves.extend(tower_of_hanoi(n-1, auxiliary, target, source))
    
    return moves

# Solve for 5 disks
moves = tower_of_hanoi(5, 0, 2, 1)

print(f"Total number of moves: {len(moves)}")
print("\nSequence of moves:")
for i, move in enumerate(moves, 1):
    disk_id, from_peg, to_peg = move
    print(f"{i:2d}. Move disk {disk_id} from peg {from_peg} to peg {to_peg}")

print(f"\nComplete moves list:")
print(moves)