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
        moves: List to store moves [disk_id, from_peg, to_peg]
    """
    if n > 0:
        # Move n-1 disks from source to auxiliary
        hanoi(n-1, source, auxiliary, target, moves)
        
        # Move the nth disk (largest) from source to target
        moves.append([n, source, target])
        
        # Move n-1 disks from auxiliary to target
        hanoi(n-1, auxiliary, target, source, moves)

def print_configuration(peg0, peg1, peg2, move_num):
    """Print current configuration."""
    print(f"\nAfter move {move_num}:")
    print(f"Peg 0 (bottom to top): {peg0}")
    print(f"Peg 1 (bottom to top): {peg1}")
    print(f"Peg 2 (bottom to top): {peg2}")

def simulate_moves(moves):
    """Simulate the moves to verify correctness."""
    # Initialize pegs
    peg0 = [5, 4, 3, 2, 1]  # bottom to top
    peg1 = []
    peg2 = []
    
    print("Initial configuration:")
    print(f"Peg 0 (bottom to top): {peg0}")
    print(f"Peg 1 (bottom to top): {peg1}")
    print(f"Peg 2 (bottom to top): {peg2}")
    
    for i, move in enumerate(moves):
        disk_id, from_peg, to_peg = move
        
        # Get the source and target pegs
        if from_peg == 0:
            source_peg = peg0
        elif from_peg == 1:
            source_peg = peg1
        else:
            source_peg = peg2
            
        if to_peg == 0:
            target_peg = peg0
        elif to_peg == 1:
            target_peg = peg1
        else:
            target_peg = peg2
        
        # Check if move is valid
        if not source_peg:
            raise ValueError(f"Move {i+1}: Cannot move disk {disk_id} from empty peg {from_peg}")
        
        # The top disk is the last element (top of stack)
        top_disk = source_peg[-1]
        if top_disk != disk_id:
            raise ValueError(f"Move {i+1}: Disk {disk_id} is not on top of peg {from_peg}. Top disk is {top_disk}")
        
        # Check if target peg is empty or has larger disk
        if target_peg and target_peg[-1] < disk_id:
            raise ValueError(f"Move {i+1}: Cannot place disk {disk_id} on top of smaller disk {target_peg[-1]}")
        
        # Make the move
        source_peg.pop()
        target_peg.append(disk_id)
        
        print_configuration(peg0, peg1, peg2, i+1)
    
    # Check final state
    if peg0 == [] and peg1 == [] and peg2 == [5, 4, 3, 2, 1]:
        print("\n✓ SUCCESS: Final configuration matches goal!")
        return True
    else:
        print(f"\n✗ FAILURE: Final configuration does not match goal!")
        print(f"Expected: Peg 0: [], Peg 1: [], Peg 2: [5, 4, 3, 2, 1]")
        print(f"Actual:   Peg 0: {peg0}, Peg 1: {peg1}, Peg 2: {peg2}")
        return False

# Generate the solution
moves = []
hanoi(5, 0, 2, 1, moves)

print("=" * 60)
print("TOWER OF HANOI - 5 DISKS")
print("=" * 60)
print(f"\nNumber of moves required: {len(moves)}")
print(f"Expected (2^5 - 1): {2**5 - 1}")

print("\n" + "=" * 60)
print("COMPLETE LIST OF MOVES")
print("=" * 60)
for i, move in enumerate(moves, 1):
    disk_id, from_peg, to_peg = move
    print(f"Move {i:2d}: Disk {disk_id} from peg {from_peg} to peg {to_peg}")

print("\n" + "=" * 60)
print("SIMULATING MOVES TO VERIFY CORRECTNESS")
print("=" * 60)
success = simulate_moves(moves)

if success:
    print("\n" + "=" * 60)
    print("FINAL ANSWER")
    print("=" * 60)
    print("moves = [")
    for i, move in enumerate(moves):
        if i < len(moves) - 1:
            print(f"    {move},")
        else:
            print(f"    {move}")
    print("]")
else:
    print("\nError in solution!")