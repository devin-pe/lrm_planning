# Tool: write_file
# Target path: /home/user/hanoi_solver.py
# Step: 2

#!/usr/bin/env python3
"""
Tower of Hanoi solver for 5 disks.
Generates the optimal sequence of moves from peg 0 to peg 2.
"""

def hanoi(n, source, target, auxiliary, moves):
    """
    Recursive Tower of Hanoi solution.
    
    Args:
        n: Number of disks
        source: Source peg
        target: Target peg
        auxiliary: Auxiliary peg
        moves: List to store moves
    """
    if n == 1:
        # Base case: move a single disk
        moves.append([1, source, target])
    else:
        # Move n-1 disks from source to auxiliary
        hanoi(n-1, source, auxiliary, target, moves)
        # Move the largest disk from source to target
        moves.append([n, source, target])
        # Move n-1 disks from auxiliary to target
        hanoi(n-1, auxiliary, target, source, moves)

def generate_moves(n_disks):
    """Generate all moves for n disks from peg 0 to peg 2."""
    moves = []
    hanoi(n_disks, 0, 2, 1, moves)
    return moves

def verify_solution(moves, n_disks):
    """Verify that the moves produce the correct final state."""
    # Initialize pegs
    pegs = [list(range(n_disks, 0, -1)), [], []]
    
    print(f"Initial state: {pegs}")
    print(f"Number of moves: {len(moves)}")
    print("\nExecuting moves...\n")
    
    for i, move in enumerate(moves, 1):
        disk_id, from_peg, to_peg = move
        
        # Validate move
        if not pegs[from_peg]:
            print(f"ERROR: Move {i}: Peg {from_peg} is empty!")
            return False
        
        top_disk = pegs[from_peg][-1]
        if top_disk != disk_id:
            print(f"ERROR: Move {i}: Trying to move disk {disk_id} but top disk is {top_disk}!")
            return False
        
        # Check if target peg is empty or has larger disk
        if pegs[to_peg] and pegs[to_peg][-1] < disk_id:
            print(f"ERROR: Move {i}: Cannot place disk {disk_id} on top of disk {pegs[to_peg][-1]}!")
            return False
        
        # Execute move
        pegs[from_peg].pop()
        pegs[to_peg].append(disk_id)
        
        print(f"Move {i:2d}: Disk {disk_id} from peg {from_peg} to peg {to_peg} -> {pegs}")
    
    # Check final state
    expected = [[], [], list(range(n_disks, 0, -1))]
    if pegs == expected:
        print(f"\n✓ SUCCESS! Final state matches goal: {pegs}")
        return True
    else:
        print(f"\n✗ FAILURE! Final state {pegs} does not match goal {expected}")
        return False

# Generate and verify solution for 5 disks
if __name__ == "__main__":
    n_disks = 5
    moves = generate_moves(n_disks)
    
    print("=" * 60)
    print(f"TOWER OF HANOI - {n_disks} DISKS")
    print("=" * 60)
    print(f"\nOptimal solution requires {2**n_disks - 1} moves.")
    print(f"Generated {len(moves)} moves.\n")
    
    # Display all moves
    print("Complete list of moves:")
    print("moves = [")
    for i, move in enumerate(moves, 1):
        print(f"    {move},  # Move {i}")
    print("]")
    
    # Verify the solution
    print("\n" + "=" * 60)
    print("VERIFYING SOLUTION...")
    print("=" * 60)
    success = verify_solution(moves, n_disks)
    
    if success:
        print("\n" + "=" * 60)
        print("SOLUTION VERIFIED!")
        print("=" * 60)
        print(f"\nThe optimal sequence of {len(moves)} moves is:")
        print("moves = [")
        for move in moves:
            print(f"    {move},")
        print("]")