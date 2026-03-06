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
    if n > 0:
        # Move n-1 disks from source to auxiliary
        hanoi(n-1, source, auxiliary, target, moves)
        
        # Move the nth disk from source to target
        moves.append([n, source, target])
        
        # Move n-1 disks from auxiliary to target
        hanoi(n-1, auxiliary, target, source, moves)

def print_state(pegs, move_num=None):
    """Print current state of pegs."""
    if move_num is not None:
        print(f"\nAfter move {move_num}:")
    for i, peg in enumerate(pegs):
        print(f"Peg {i}: {peg}")

def simulate_moves(moves, n):
    """Simulate the moves to verify correctness."""
    # Initialize pegs: peg 0 has all disks, others are empty
    pegs = {0: list(range(n, 0, -1)), 1: [], 2: []}
    
    print("Initial state:")
    print_state(pegs)
    
    for i, (disk, from_peg, to_peg) in enumerate(moves, 1):
        # Verify the move is valid
        if not pegs[from_peg] or pegs[from_peg][-1] != disk:
            raise ValueError(f"Invalid move {i}: Disk {disk} not on top of peg {from_peg}")
        
        if pegs[to_peg] and pegs[to_peg][-1] < disk:
            raise ValueError(f"Invalid move {i}: Cannot place disk {disk} on smaller disk {pegs[to_peg][-1]}")
        
        # Make the move
        pegs[from_peg].pop()
        pegs[to_peg].append(disk)
        
        print_state(pegs, i)
    
    # Verify final state
    if pegs[0] or pegs[1] or pegs[2] != list(range(n, 0, -1)):
        raise ValueError("Final state does not match goal configuration")
    
    print("\n✓ All moves are valid and goal configuration reached!")

# Solve for 5 disks
if __name__ == "__main__":
    n = 5
    moves = []
    
    # Generate moves: move 5 disks from peg 0 to peg 2 using peg 1 as auxiliary
    hanoi(n, 0, 2, 1, moves)
    
    print(f"Solution for {n} disks requires {len(moves)} moves:")
    print("\nmoves = [")
    for i, move in enumerate(moves):
        print(f"    {move},  # Move {i+1}")
    print("]")
    
    # Verify the solution
    print("\n" + "="*60)
    print("Verifying moves...")
    print("="*60)
    simulate_moves(moves, n)