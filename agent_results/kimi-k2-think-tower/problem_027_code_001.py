# Tool: write_file
# Target path: /home/user/hanoi_solver.py
# Step: 2

#!/usr/bin/env python3
"""
Tower of Hanoi Solver
Solves the puzzle for n disks and returns the sequence of moves.
"""

def hanoi(n, source, target, auxiliary, moves=None):
    """
    Solve Tower of Hanoi puzzle recursively.
    
    Args:
        n: Number of disks
        source: Source peg (0, 1, or 2)
        target: Target peg (0, 1, or 2)
        auxiliary: Auxiliary peg (0, 1, or 2)
        moves: List to store moves
    
    Returns:
        List of moves in format [[disk_id, from_peg, to_peg], ...]
    """
    if moves is None:
        moves = []
    
    if n == 1:
        # Base case: move single disk
        moves.append([1, source, target])
        return moves
    
    # Move n-1 disks from source to auxiliary
    hanoi(n-1, source, auxiliary, target, moves)
    
    # Move disk n from source to target
    moves.append([n, source, target])
    
    # Move n-1 disks from auxiliary to target
    hanoi(n-1, auxiliary, target, source, moves)
    
    return moves

def simulate_moves(moves, n):
    """
    Simulate the moves to verify correctness.
    """
    # Initialize pegs
    pegs = [list(range(n, 0, -1)), [], []]
    
    print(f"Initial state: {pegs}")
    
    for i, move in enumerate(moves):
        disk_id, from_peg, to_peg = move
        
        # Validate move
        if not pegs[from_peg]:
            raise ValueError(f"Move {i+1}: Peg {from_peg} is empty!")
        
        if pegs[from_peg][-1] != disk_id:
            raise ValueError(f"Move {i+1}: Disk {disk_id} is not on top of peg {from_peg}!")
        
        if pegs[to_peg] and pegs[to_peg][-1] < disk_id:
            raise ValueError(f"Move {i+1}: Cannot place larger disk {disk_id} on smaller disk {pegs[to_peg][-1]}!")
        
        # Execute move
        disk = pegs[from_peg].pop()
        pegs[to_peg].append(disk)
        
        print(f"Move {i+1:2d}: Disk {disk_id} from peg {from_peg} to peg {to_peg} -> {pegs}")
    
    # Verify final state
    if pegs[0] == [] and pegs[1] == [] and pegs[2] == list(range(n, 0, -1)):
        print(f"\nSuccess! Final state matches goal: {pegs}")
        return True
    else:
        print(f"\nFailed! Final state: {pegs}")
        return False

# Solve for 5 disks
print("Solving Tower of Hanoi with 5 disks...")
moves = hanoi(5, 0, 2, 1)

print(f"\nTotal moves: {len(moves)}")
print(f"Expected moves: {2**5 - 1}")

print("\n" + "="*60)
print("Sequence of moves:")
print("="*60)
for i, move in enumerate(moves, 1):
    print(f"Move {i:2d}: Disk {move[0]} from peg {move[1]} to peg {move[2]}")

print("\n" + "="*60)
print("Simulating moves to verify correctness...")
print("="*60)
success = simulate_moves(moves, 5)

print("\n" + "="*60)
print("Final Answer:")
print("="*60)
print(f"moves = {moves}")