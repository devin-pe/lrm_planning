# Tool: write_file
# Target path: /home/user/hanoi_solver.py
# Step: 2

def solve_hanoi(n, source, target, auxiliary):
    """
    Solve Tower of Hanoi puzzle recursively.
    Returns list of moves in format [disk_id, from_peg, to_peg]
    """
    moves = []
    
    if n == 1:
        # Base case: move single disk
        moves.append([1, source, target])
        return moves
    
    # Move n-1 disks from source to auxiliary
    moves.extend(solve_hanoi(n-1, source, auxiliary, target))
    
    # Move disk n from source to target
    moves.append([n, source, target])
    
    # Move n-1 disks from auxiliary to target
    moves.extend(solve_hanoi(n-1, auxiliary, target, source))
    
    return moves

def apply_moves(initial_state, moves):
    """
    Apply a sequence of moves to an initial state and return final state.
    State is represented as list of 3 pegs, each containing a list of disks (bottom to top)
    """
    # Create deep copy of initial state
    state = [peg.copy() for peg in initial_state]
    
    for move in moves:
        disk_id, from_peg, to_peg = move
        
        # Verify the move is valid
        if not state[from_peg]:
            raise ValueError(f"Invalid move: peg {from_peg} is empty")
        
        top_disk = state[from_peg][-1]
        if top_disk != disk_id:
            raise ValueError(f"Invalid move: disk {disk_id} is not on top of peg {from_peg}")
        
        if state[to_peg] and state[to_peg][-1] < disk_id:
            raise ValueError(f"Invalid move: cannot place disk {disk_id} on top of smaller disk {state[to_peg][-1]}")
        
        # Apply the move
        state[from_peg].pop()
        state[to_peg].append(disk_id)
    
    return state

# Solve for 3 disks
print("Solving Tower of Hanoi with 3 disks...")
moves = solve_hanoi(3, 0, 2, 1)

print(f"\nNumber of moves: {len(moves)}")
print("\nSequence of moves:")
for i, move in enumerate(moves, 1):
    disk_id, from_peg, to_peg = move
    print(f"{i}. Move disk {disk_id} from peg {from_peg} to peg {to_peg}")

# Verify the solution
initial_state = [[3, 2, 1], [], []]
goal_state = [[], [], [3, 2, 1]]

print(f"\nInitial state: {initial_state}")
print(f"Goal state: {goal_state}")

final_state = apply_moves(initial_state, moves)
print(f"Final state: {final_state}")

if final_state == goal_state:
    print("\n✓ Solution verified successfully!")
    print("\nComplete list of moves:")
    print(f"moves = {moves}")
else:
    print("\n✗ Solution failed to reach goal state")
    print(f"Expected: {goal_state}")
    print(f"Got:      {final_state}")