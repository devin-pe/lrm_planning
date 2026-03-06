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
    Recursive Tower of Hanoi solver.
    
    Args:
        n: Number of disks to move
        source: Source peg (0, 1, or 2)
        target: Target peg (0, 1, or 2)
        auxiliary: Auxiliary peg (0, 1, or 2)
        moves: List to store the moves
    """
    if n > 0:
        # Move n-1 disks from source to auxiliary (using target as auxiliary)
        hanoi(n-1, source, auxiliary, target, moves)
        
        # Move disk n from source to target
        moves.append([n, source, target])
        
        # Move n-1 disks from auxiliary to target (using source as auxiliary)
        hanoi(n-1, auxiliary, target, source, moves)


def solve_hanoi_5_disks():
    """Solve Tower of Hanoi with 5 disks."""
    moves = []
    hanoi(5, 0, 2, 1, moves)
    return moves


def print_solution(moves):
    """Print the solution in a readable format."""
    print(f"Solution with {len(moves)} moves:")
    print("moves = [")
    for i, move in enumerate(moves):
        disk, from_peg, to_peg = move
        print(f"    [{disk}, {from_peg}, {to_peg}],  # Move {i+1}: Disk {disk} from peg {from_peg} to peg {to_peg}")
    print("]")


def verify_solution(moves):
    """Verify the solution by simulating the moves."""
    # Initialize pegs
    pegs = {0: [5, 4, 3, 2, 1], 1: [], 2: []}
    
    print("\nVerifying solution:")
    print(f"Initial state: {pegs}")
    
    for i, move in enumerate(moves):
        disk, from_peg, to_peg = move
        
        # Check if move is valid
        if not pegs[from_peg]:
            print(f"ERROR at move {i+1}: Peg {from_peg} is empty!")
            return False
        
        top_disk = pegs[from_peg][-1]
        if top_disk != disk:
            print(f"ERROR at move {i+1}: Top disk on peg {from_peg} is {top_disk}, not {disk}!")
            return False
        
        if pegs[to_peg] and pegs[to_peg][-1] < disk:
            print(f"ERROR at move {i+1}: Cannot place disk {disk} on top of disk {pegs[to_peg][-1]}!")
            return False
        
        # Execute move
        pegs[from_peg].pop()
        pegs[to_peg].append(disk)
        
        print(f"After move {i+1:2d} (disk {disk} from {from_peg} to {to_peg}): {pegs}")
    
    # Check final state
    if pegs[0] == [] and pegs[1] == [] and pegs[2] == [5, 4, 3, 2, 1]:
        print("\n✓ Solution verified successfully!")
        return True
    else:
        print(f"\n✗ Final state is incorrect: {pegs}")
        return False


if __name__ == "__main__":
    # Generate solution
    moves = solve_hanoi_5_disks()
    
    # Print solution
    print_solution(moves)
    
    # Verify solution
    verify_solution(moves)