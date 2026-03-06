# Tool: write_file
# Target path: /home/user/hanoi_solver.py
# Step: 2

#!/usr/bin/env python3
"""
Tower of Hanoi solver for 4 disks.
Generates the optimal sequence of moves from initial to goal configuration.
"""

def hanoi(n, source, target, auxiliary, moves):
    """
    Recursive Tower of Hanoi solver.
    
    Args:
        n: Number of disks
        source: Source peg (0, 1, or 2)
        target: Target peg (0, 1, or 2)
        auxiliary: Auxiliary peg (0, 1, or 2)
        moves: List to store moves [disk_id, from_peg, to_peg]
    """
    if n > 0:
        # Move n-1 disks from source to auxiliary
        hanoi(n - 1, source, auxiliary, target, moves)
        
        # Move the nth disk from source to target
        moves.append([n, source, target])
        
        # Move n-1 disks from auxiliary to target
        hanoi(n - 1, auxiliary, target, source, moves)

def simulate_moves(moves, n):
    """
    Simulate the moves to verify correctness.
    
    Args:
        moves: List of moves [disk_id, from_peg, to_peg]
        n: Number of disks
    """
    # Initialize pegs: each peg is a list with bottom disks first
    pegs = [list(range(n, 0, -1)), [], []]
    
    print("Initial state:")
    for i, peg in enumerate(pegs):
        print(f"  Peg {i}: {peg}")
    print()
    
    for i, (disk, from_peg, to_peg) in enumerate(moves, 1):
        # Validate move
        if not pegs[from_peg] or pegs[from_peg][-1] != disk:
            print(f"ERROR: Move {i}: Invalid move - disk {disk} not on top of peg {from_peg}")
            return False
        
        if pegs[to_peg] and pegs[to_peg][-1] < disk:
            print(f"ERROR: Move {i}: Invalid move - cannot place disk {disk} on top of smaller disk {pegs[to_peg][-1]}")
            return False
        
        # Execute move
        pegs[from_peg].pop()
        pegs[to_peg].append(disk)
        
        print(f"Move {i}: Disk {disk} from peg {from_peg} to peg {to_peg}")
        for j, peg in enumerate(pegs):
            print(f"  Peg {j}: {peg}")
        print()
    
    # Check final state
    if pegs[0] == [] and pegs[1] == [] and pegs[2] == list(range(n, 0, -1)):
        print("SUCCESS: Goal configuration reached!")
        return True
    else:
        print("FAILURE: Did not reach goal configuration")
        print(f"Final state: {pegs}")
        return False

# Solve for 4 disks
if __name__ == "__main__":
    n = 4
    moves = []
    
    # Solve Tower of Hanoi: move 4 disks from peg 0 to peg 2 using peg 1 as auxiliary
    hanoi(n, 0, 2, 1, moves)
    
    print(f"Generated {len(moves)} moves for {n} disks:")
    print("moves = [")
    for move in moves:
        print(f"    {move},")
    print("]")
    print()
    
    # Verify the solution
    print("Simulating moves to verify correctness...")
    print("=" * 60)
    success = simulate_moves(moves, n)
    print("=" * 60)
    
    if success:
        print(f"\nOptimal solution found with {len(moves)} moves!")
    else:
        print("\nSolution verification failed!")