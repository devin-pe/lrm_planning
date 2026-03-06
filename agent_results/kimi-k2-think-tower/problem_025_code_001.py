# Tool: write_file
# Target path: /home/user/hanoi_solver.py
# Step: 2

#!/usr/bin/env python3
"""
Tower of Hanoi solver for 5 disks
Initial: Peg 0 has [5,4,3,2,1], Pegs 1 and 2 are empty
Goal: Peg 2 has [5,4,3,2,1], Pegs 0 and 1 are empty
"""

def tower_of_hanoi(n, source, target, auxiliary, moves):
    """
    Solve Tower of Hanoi puzzle recursively
    
    Args:
        n: Number of disks
        source: Source peg (0, 1, or 2)
        target: Target peg (0, 1, or 2)
        auxiliary: Auxiliary peg (0, 1, or 2)
        moves: List to store moves [disk_id, from_peg, to_peg]
    """
    if n == 1:
        # Base case: move single disk
        moves.append([1, source, target])
    else:
        # Move n-1 disks from source to auxiliary
        tower_of_hanoi(n-1, source, auxiliary, target, moves)
        # Move the nth disk (largest) from source to target
        moves.append([n, source, target])
        # Move n-1 disks from auxiliary to target
        tower_of_hanoi(n-1, auxiliary, target, source, moves)

def simulate_moves(initial_state, moves):
    """
    Simulate the moves to verify correctness
    """
    # Create a copy of the initial state
    pegs = [list(peg) for peg in initial_state]
    
    print("Initial state:")
    for i, peg in enumerate(pegs):
        print(f"Peg {i}: {peg}")
    print()
    
    for i, move in enumerate(moves):
        disk_id, from_peg, to_peg = move
        
        # Validate move
        if not pegs[from_peg]:
            raise ValueError(f"Move {i+1}: Peg {from_peg} is empty!")
        
        if pegs[from_peg][-1] != disk_id:
            raise ValueError(f"Move {i+1}: Top disk on peg {from_peg} is {pegs[from_peg][-1]}, not {disk_id}!")
        
        if pegs[to_peg] and pegs[to_peg][-1] < disk_id:
            raise ValueError(f"Move {i+1}: Cannot place disk {disk_id} on top of smaller disk {pegs[to_peg][-1]}!")
        
        # Execute move
        pegs[from_peg].pop()
        pegs[to_peg].append(disk_id)
        
        print(f"Move {i+1:2d}: Disk {disk_id} from peg {from_peg} to peg {to_peg}")
        print(f"         State: Peg 0: {pegs[0]}, Peg 1: {pegs[1]}, Peg 2: {pegs[2]}")
    
    print("\nFinal state:")
    for i, peg in enumerate(pegs):
        print(f"Peg {i}: {peg}")
    
    return pegs

# Main execution
if __name__ == "__main__":
    # Initial configuration
    # Peg 0 has disks 5 (bottom) to 1 (top)
    initial_state = [[5, 4, 3, 2, 1], [], []]
    
    # Generate solution
    moves = []
    tower_of_hanoi(5, 0, 2, 1, moves)
    
    print("Solution found!")
    print(f"Number of moves: {len(moves)}")
    print(f"Expected moves for 5 disks: {2**5 - 1}")
    print()
    
    # Display moves in required format
    print("moves = [")
    for i, move in enumerate(moves):
        print(f"    {move},", end="")
        if i < len(moves) - 1:
            print()
    print("\n]")
    print()
    
    # Verify by simulation
    print("=" * 60)
    print("SIMULATING MOVES TO VERIFY CORRECTNESS")
    print("=" * 60)
    final_state = simulate_moves(initial_state, moves)
    
    # Check if goal is reached
    goal_state = [[], [], [5, 4, 3, 2, 1]]
    if final_state == goal_state:
        print("\n✓ SUCCESS! Goal configuration reached!")
    else:
        print(f"\n✗ FAILED! Final state does not match goal.")
        print(f"Expected: {goal_state}")
        print(f"Got:      {final_state}")