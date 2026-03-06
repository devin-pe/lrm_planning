# Tool: execute_python
# Step: 2

def tower_of_hanoi(n, source, target, auxiliary):
    """
    Solve Tower of Hanoi puzzle and return list of moves.
    Each move is [disk_id, from_peg, to_peg]
    """
    moves = []
    
    def move_disks(num_disks, src, tgt, aux):
        if num_disks == 0:
            return
        
        # Move n-1 disks from source to auxiliary
        move_disks(num_disks - 1, src, aux, tgt)
        
        # Move the nth disk from source to target
        moves.append([num_disks, src, tgt])
        
        # Move n-1 disks from auxiliary to target
        move_disks(num_disks - 1, aux, tgt, src)
    
    move_disks(n, source, target, auxiliary)
    return moves

# Solve for 4 disks from peg 0 to peg 2 using peg 1 as auxiliary
moves = tower_of_hanoi(4, 0, 2, 1)
moves
