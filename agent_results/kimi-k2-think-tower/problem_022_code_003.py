# Tool: write_file
# Target path: /home/user/hanoi_solver_fixed.py
# Step: 6

#!/usr/bin/env python3
"""
Tower of Hanoi solver for 5 disks with proper state visualization.
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

def simulate_and_show(moves, n):
    """Simulate moves with proper visualization."""
    # Initialize pegs: peg 0 has all disks (5 at bottom, 1 at top)
    pegs = {0: list(range(n, 0, -1)), 1: [], 2: []}
    
    print("="*70)
    print("INITIAL STATE")
    print("="*70)
    for i in range(3):
        print(f"Peg {i}: {pegs[i]}")
    
    for i, (disk, from_peg, to_peg) in enumerate(moves, 1):
        print(f"\n{'='*70}")
        print(f"MOVE {i}: Disk {disk} from peg {from_peg} → peg {to_peg}")
        print(f"{'='*70}")
        
        # Verify and make the move
        if not pegs[from_peg] or pegs[from_peg][-1] != disk:
            raise ValueError(f"Invalid move {i}: Disk {disk} not on top of peg {from_peg}")
        
        if pegs[to_peg] and pegs[to_peg][-1] < disk:
            raise ValueError(f"Invalid move {i}: Cannot place disk {disk} on smaller disk {pegs[to_peg][-1]}")
        
        # Make the move
        pegs[from_peg].pop()
        pegs[to_peg].append(disk)
        
        # Show current state
        for peg in range(3):
            print(f"Peg {peg}: {pegs[peg]}")
    
    # Verify final state
    print(f"\n{'='*70}")
    print("FINAL STATE")
    print("="*70)
    for i in range(3):
        print(f"Peg {i}: {pegs[i]}")
    
    if pegs[0] or pegs[1] or pegs[2] != list(range(n, 0, -1)):
        raise ValueError("Final state does not match goal configuration")
    
    print("\n✓ SUCCESS! All moves are valid and goal configuration reached!")
    print(f"✓ Total moves: {len(moves)}")

# Solve for 5 disks
if __name__ == "__main__":
    n = 5
    moves = []
    
    # Generate moves: move 5 disks from peg 0 to peg 2 using peg 1 as auxiliary
    hanoi(n, 0, 2, 1, moves)
    
    print(f"Tower of Hanoi solution for {n} disks")
    print(f"Optimal solution requires {len(moves)} moves\n")
    
    print("Complete list of moves:")
    print("moves = [")
    for i, move in enumerate(moves, 1):
        disk, from_peg, to_peg = move
        print(f"    {move},  # Move {i}: Disk {disk} from peg {from_peg} to peg {to_peg}")
    print("]")
    
    # Simulate and verify
    print("\n" + "="*70)
    print("SIMULATING MOVES TO VERIFY CORRECTNESS")
    print("="*70)
    simulate_and_show(moves, n)