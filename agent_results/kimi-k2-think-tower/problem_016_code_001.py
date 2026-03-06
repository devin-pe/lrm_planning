# Tool: write_file
# Target path: /home/user/hanoi_solver.py
# Step: 2

#!/usr/bin/env python3
"""
Tower of Hanoi solver for 4 disks.
Simulates the moves to verify correctness.
"""

def solve_hanoi(n, source, target, auxiliary):
    """
    Recursive solution for Tower of Hanoi.
    Returns list of moves [disk, from, to].
    """
    moves = []
    
    if n == 1:
        # Base case: move single disk
        moves.append([1, source, target])
        return moves
    
    # Move n-1 disks from source to auxiliary
    moves.extend(solve_hanoi(n-1, source, auxiliary, target))
    
    # Move nth disk from source to target
    moves.append([n, source, target])
    
    # Move n-1 disks from auxiliary to target
    moves.extend(solve_hanoi(n-1, auxiliary, target, source))
    
    return moves

def simulate_moves(initial_state, moves):
    """
    Simulate the moves on the pegs to verify correctness.
    """
    # Create a copy of the initial state
    pegs = [list(peg) for peg in initial_state]
    
    print("Initial state:")
    for i, peg in enumerate(pegs):
        print(f"Peg {i}: {peg}")
    print()
    
    for i, move in enumerate(moves, 1):
        disk, from_peg, to_peg = move
        
        # Validate move
        if not pegs[from_peg]:
            raise ValueError(f"Move {i}: Peg {from_peg} is empty!")
        
        if pegs[from_peg][-1] != disk:
            raise ValueError(f"Move {i}: Disk {disk} is not on top of peg {from_peg}!")
        
        if pegs[to_peg] and pegs[to_peg][-1] < disk:
            raise ValueError(f"Move {i}: Cannot place disk {disk} on top of smaller disk {pegs[to_peg][-1]}!")
        
        # Execute move
        pegs[from_peg].pop()
        pegs[to_peg].append(disk)
        
        print(f"Move {i}: Disk {disk} from peg {from_peg} to peg {to_peg}")
        for j, peg in enumerate(pegs):
            print(f"  Peg {j}: {peg}")
        print()
    
    return pegs

def main():
    # Initial configuration
    # Peg 0: [4, 3, 2, 1] (bottom to top)
    # Peg 1: []
    # Peg 2: []
    initial_state = [[4, 3, 2, 1], [], []]
    
    # Goal configuration
    # Peg 0: []
    # Peg 1: []
    # Peg 2: [4, 3, 2, 1]
    goal_state = [[], [], [4, 3, 2, 1]]
    
    # Generate optimal solution (2^4 - 1 = 15 moves)
    moves = solve_hanoi(4, 0, 2, 1)
    
    print("=" * 60)
    print("TOWER OF HANOI - 4 DISKS")
    print("=" * 60)
    print(f"Number of moves: {len(moves)}")
    print(f"Expected: {2**4 - 1} moves")
    print()
    
    # Simulate the moves
    final_state = simulate_moves(initial_state, moves)
    
    # Verify final state
    print("=" * 60)
    print("VERIFICATION")
    print("=" * 60)
    
    if final_state == goal_state:
        print("✓ SUCCESS! Final state matches goal configuration.")
    else:
        print("✗ FAILURE! Final state does not match goal.")
        print(f"Expected: {goal_state}")
        print(f"Actual:   {final_state}")
    
    print()
    print("Complete list of moves:")
    print("moves = [")
    for i, move in enumerate(moves):
        if i < len(moves) - 1:
            print(f"    {move},")
        else:
            print(f"    {move}")
    print("]")

if __name__ == "__main__":
    main()