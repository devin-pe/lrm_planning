# Tool: execute_python
# Step: 2


def tower_of_hanoi(n, source, target, auxiliary):
    """
    Generate moves for Tower of Hanoi puzzle.
    
    Args:
        n: Number of disks
        source: Source peg
        target: Target peg
        auxiliary: Auxiliary peg
    
    Returns:
        List of moves in format [disk_id, from_peg, to_peg]
    """
    moves = []
    
    if n == 1:
        # Base case: move single disk
        moves.append([1, source, target])
        return moves
    
    # Move n-1 disks from source to auxiliary
    moves.extend(tower_of_hanoi(n-1, source, auxiliary, target))
    
    # Move the nth disk from source to target
    moves.append([n, source, target])
    
    # Move n-1 disks from auxiliary to target
    moves.extend(tower_of_hanoi(n-1, auxiliary, target, source))
    
    return moves

# Generate moves for 4 disks from peg 0 to peg 2 using peg 1 as auxiliary
moves = tower_of_hanoi(4, 0, 2, 1)

print(f"Total number of moves: {len(moves)}")
print("\nSequence of moves:")
for i, move in enumerate(moves, 1):
    disk_id, from_peg, to_peg = move
    print(f"{i:2d}. Move disk {disk_id} from peg {from_peg} to peg {to_peg}")

print(f"\nmoves = {moves}")
