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

def print_state(pegs, move_num=None):
    """Print the current state of the pegs."""
    if move_num is not None:
        print(f"\nAfter move {move_num}:")
    for i, peg in enumerate(pegs):
        print(f"Peg {i}: {peg}")

def simulate_moves(moves, n_disks):
    """
    Simulate the moves to verify correctness.
    Returns True if the simulation reaches the goal state.
    """
    # Initialize pegs: peg 0 has all disks (largest at bottom)
    pegs = [list(range(n_disks, 0, -1)), [], []]
    
    print("Initial state:")
    print_state(pegs)
    
    # Simulate each move
    for i, (disk_id, from_peg, to_peg) in enumerate(moves, 1):
        # Validate the move
        if not pegs[from_peg]:
            raise ValueError(f"Move {i}: Peg {from_peg} is empty!")
        
        top_disk = pegs[from_peg][-1]
        if top_disk != disk_id:
            raise ValueError(f"Move {i}: Trying to move disk {disk_id} but top disk is {top_disk}")
        
        if pegs[to_peg] and pegs[to_peg][-1] < disk_id:
            raise ValueError(f"Move {i}: Cannot place disk {disk_id} on smaller disk {pegs[to_peg][-1]}")
        
        # Execute the move
        pegs[from_peg].pop()
        pegs[to_peg].append(disk_id)
        
        if i % 5 == 0:  # Print every 5 moves to avoid too much output
            print_state(pegs, i)
    
    print("\nFinal state:")
    print_state(pegs, len(moves))
    
    # Check if goal reached
    goal_state = [[], [], list(range(n_disks, 0, -1))]
    return pegs == goal_state

# Main execution
if __name__ == "__main__":
    n_disks = 5
    
    print("=" * 60)
    print(f"Tower of Hanoi with {n_disks} disks")
    print("Initial: Peg 0 has [5, 4, 3, 2, 1], Pegs 1 and 2 are empty")
    print("Goal: Peg 2 has [5, 4, 3, 2, 1], Pegs 0 and 1 are empty")
    print("=" * 60)
    
    # Generate moves
    moves = generate_moves(n_disks)
    
    print(f"\nTotal moves required: {len(moves)}")
    print(f"Expected: {2**n_disks - 1} moves (2^n - 1)")
    
    # Display the moves
    print("\nMoves (format: [disk_id, from_peg, to_peg]):")
    for i, move in enumerate(moves, 1):
        print(f"{i:2d}: {move}")
    
    # Verify by simulation
    print("\n" + "=" * 60)
    print("Simulating moves to verify correctness...")
    print("=" * 60)
    
    success = simulate_moves(moves, n_disks)
    
    if success:
        print("\n✓ SUCCESS! All moves are valid and goal state reached.")
    else:
        print("\n✗ FAILED! Goal state not reached.")
    
    # Print the final answer in required format
    print("\n" + "=" * 60)
    print("FINAL ANSWER:")
    print("=" * 60)
    print("moves = [")
    for i, move in enumerate(moves):
        if i < len(moves) - 1:
            print(f"    {move},")
        else:
            print(f"    {move}")
    print("]")