# Tool: write_file
# Target path: /home/user/hanoi_solver.py
# Step: 2

#!/usr/bin/env python3
"""
Tower of Hanoi Solver for 5 disks
Generates the optimal sequence of moves from initial to goal configuration
"""

def tower_of_hanoi(n, source, target, auxiliary, moves):
    """
    Recursive function to solve Tower of Hanoi
    
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
        return
    
    # Move n-1 disks from source to auxiliary peg
    tower_of_hanoi(n-1, source, auxiliary, target, moves)
    
    # Move the nth disk (largest) from source to target
    moves.append([n, source, target])
    
    # Move n-1 disks from auxiliary to target peg
    tower_of_hanoi(n-1, auxiliary, target, source, moves)


def generate_solution(n):
    """Generate solution for n disks"""
    moves = []
    tower_of_hanoi(n, 0, 2, 1, moves)
    return moves


def print_solution(moves):
    """Print the solution in a readable format"""
    print(f"Solution for {len(set(move[0] for move in moves))} disks:")
    print(f"Total moves: {len(moves)}")
    print("\nMoves:")
    for i, move in enumerate(moves, 1):
        disk, from_peg, to_peg = move
        print(f"{i:2d}. Move disk {disk} from peg {from_peg} to peg {to_peg}")
    print("\nPython list format:")
    print(f"moves = {moves}")


def verify_solution(moves, n):
    """Verify that the solution is correct"""
    # Initialize pegs
    pegs = {0: list(range(n, 0, -1)), 1: [], 2: []}
    
    print("\nVerifying solution...")
    print(f"Initial state: {pegs}")
    
    for i, move in enumerate(moves, 1):
        disk, from_peg, to_peg = move
        
        # Check if move is valid
        if not pegs[from_peg]:
            print(f"ERROR at move {i}: Peg {from_peg} is empty!")
            return False
        
        if pegs[from_peg][-1] != disk:
            print(f"ERROR at move {i}: Disk {disk} is not on top of peg {from_peg}!")
            return False
        
        if pegs[to_peg] and pegs[to_peg][-1] < disk:
            print(f"ERROR at move {i}: Cannot place disk {disk} on top of disk {pegs[to_peg][-1]}!")
            return False
        
        # Make the move
        pegs[from_peg].pop()
        pegs[to_peg].append(disk)
        
        # Print intermediate state for first few moves
        if i <= 5 or i >= len(moves) - 5:
            print(f"After move {i:2d}: {pegs}")
    
    # Check final state
    if pegs[0] == [] and pegs[1] == [] and pegs[2] == list(range(n, 0, -1)):
        print("\n✓ Solution verified successfully!")
        return True
    else:
        print(f"\n✗ ERROR: Final state is incorrect: {pegs}")
        return False


if __name__ == "__main__":
    n = 5
    print(f"Tower of Hanoi with {n} disks")
    print("=" * 50)
    
    # Generate solution
    moves = generate_solution(n)
    
    # Print solution
    print_solution(moves)
    
    # Verify solution
    verify_solution(moves, n)