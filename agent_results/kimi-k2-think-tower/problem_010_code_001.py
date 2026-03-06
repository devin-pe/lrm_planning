# Tool: write_file
# Target path: /home/user/hanoi_solver.py
# Step: 2

#!/usr/bin/env python3
"""
Tower of Hanoi Solver for 4 disks
Initial: Peg 0: [4,3,2,1], Peg 1: [], Peg 2: []
Goal: Peg 0: [], Peg 1: [], Peg 2: [4,3,2,1]
"""

def solve_hanoi(n, source, target, auxiliary):
    """
    Solve Tower of Hanoi recursively and return list of moves.
    Each move: [disk_id, from_peg, to_peg]
    """
    moves = []
    
    if n == 1:
        # Base case: move single disk
        moves.append([1, source, target])
    else:
        # Move n-1 disks from source to auxiliary
        moves.extend(solve_hanoi(n-1, source, auxiliary, target))
        
        # Move largest disk from source to target
        moves.append([n, source, target])
        
        # Move n-1 disks from auxiliary to target
        moves.extend(solve_hanoi(n-1, auxiliary, target, source))
    
    return moves

def simulate_moves(initial_state, moves):
    """Simulate the moves and return final state"""
    # Create a copy of the initial state
    pegs = [list(peg) for peg in initial_state]
    
    print("Initial state:")
    for i, peg in enumerate(pegs):
        print(f"  Peg {i}: {peg}")
    print()
    
    for idx, move in enumerate(moves, 1):
        disk_id, from_peg, to_peg = move
        
        # Validate move
        if not pegs[from_peg]:
            raise ValueError(f"Move {idx}: Peg {from_peg} is empty!")
        
        top_disk = pegs[from_peg][-1]
        if top_disk != disk_id:
            raise ValueError(f"Move {idx}: Top disk on peg {from_peg} is {top_disk}, not {disk_id}!")
        
        if pegs[to_peg] and pegs[to_peg][-1] < disk_id:
            raise ValueError(f"Move {idx}: Cannot place larger disk {disk_id} on smaller disk {pegs[to_peg][-1]}!")
        
        # Execute move
        pegs[from_peg].pop()
        pegs[to_peg].append(disk_id)
        
        print(f"Move {idx}: Disk {disk_id} from peg {from_peg} to peg {to_peg}")
        for i, peg in enumerate(pegs):
            print(f"  Peg {i}: {peg}")
        print()
    
    return pegs

# Solve for 4 disks
print("Solving Tower of Hanoi with 4 disks...")
print("=" * 50)

moves = solve_hanoi(4, 0, 2, 1)

print(f"Solution requires {len(moves)} moves:")
for i, move in enumerate(moves, 1):
    print(f"  {i}. Move disk {move[0]} from peg {move[1]} to peg {move[2]}")

print("\n" + "=" * 50)
print("Simulating moves to verify solution...")
print("=" * 50)

# Initial state
initial_state = [[4, 3, 2, 1], [], []]
final_state = simulate_moves(initial_state, moves)

print("=" * 50)
print("Final state:")
for i, peg in enumerate(final_state):
    print(f"  Peg {i}: {peg}")

# Verify goal
goal_state = [[], [], [4, 3, 2, 1]]
if final_state == goal_state:
    print("\n✓ SUCCESS! Goal configuration reached!")
else:
    print("\n✗ FAILED! Goal configuration not reached.")

print("\n" + "=" * 50)
print("Complete list of moves:")
print("=" * 50)
print("moves = [")
for move in moves:
    print(f"    {move},")
print("]")