# Tool: write_file
# Target path: /home/user/hanoi_solver.py
# Step: 2

#!/usr/bin/env python3
"""
Tower of Hanoi solver for 4 disks.
Generates the optimal sequence of moves from peg 0 to peg 2.
"""

def hanoi(n, source, target, auxiliary, moves):
    """
    Recursive Tower of Hanoi solver.
    
    Args:
        n: Number of disks
        source: Source peg
        target: Target peg
        auxiliary: Auxiliary peg
        moves: List to store moves
    """
    if n == 1:
        # Base case: move single disk
        moves.append([1, source, target])
    else:
        # Move n-1 disks from source to auxiliary
        hanoi(n-1, source, auxiliary, target, moves)
        # Move nth disk from source to target
        moves.append([n, source, target])
        # Move n-1 disks from auxiliary to target
        hanoi(n-1, auxiliary, target, source, moves)

def solve_hanoi_4_disks():
    """Solve Tower of Hanoi for 4 disks from peg 0 to peg 2."""
    moves = []
    hanoi(4, 0, 2, 1, moves)
    return moves

def print_moves(moves):
    """Print moves in a readable format."""
    print(f"Total moves: {len(moves)}")
    print("\nSequence of moves:")
    for i, move in enumerate(moves, 1):
        disk, from_peg, to_peg = move
        print(f"{i:2d}. Move disk {disk} from peg {from_peg} to peg {to_peg}")

def verify_solution(moves):
    """Verify the solution by simulating the moves."""
    # Initialize pegs: peg 0 has disks 4,3,2,1 (bottom to top)
    pegs = {0: [4, 3, 2, 1], 1: [], 2: []}
    
    print("\nInitial state:")
    for peg in range(3):
        print(f"Peg {peg}: {pegs[peg]}")
    
    for i, move in enumerate(moves, 1):
        disk, from_peg, to_peg = move
        
        # Validate move
        if not pegs[from_peg]:
            return False, f"Error at move {i}: Peg {from_peg} is empty"
        
        top_disk = pegs[from_peg][-1]
        if top_disk != disk:
            return False, f"Error at move {i}: Top disk on peg {from_peg} is {top_disk}, not {disk}"
        
        if pegs[to_peg] and pegs[to_peg][-1] < disk:
            return False, f"Error at move {i}: Cannot place disk {disk} on top of smaller disk {pegs[to_peg][-1]}"
        
        # Execute move
        pegs[from_peg].pop()
        pegs[to_peg].append(disk)
        
        print(f"\nAfter move {i} (disk {disk} from {from_peg} to {to_peg}):")
        for peg in range(3):
            print(f"Peg {peg}: {pegs[peg]}")
    
    # Check final state
    if pegs[0] == [] and pegs[1] == [] and pegs[2] == [4, 3, 2, 1]:
        return True, "Solution verified successfully!"
    else:
        return False, f"Final state incorrect: {pegs}"

if __name__ == "__main__":
    print("=" * 60)
    print("Tower of Hanoi Solver - 4 Disks")
    print("=" * 60)
    
    # Generate solution
    moves = solve_hanoi_4_disks()
    
    # Print moves
    print_moves(moves)
    
    # Verify solution
    print("\n" + "=" * 60)
    print("Verifying solution...")
    print("=" * 60)
    success, message = verify_solution(moves)
    print(message)
    
    # Print final answer
    print("\n" + "=" * 60)
    print("FINAL ANSWER:")
    print("=" * 60)
    print(f"moves = {moves}")
