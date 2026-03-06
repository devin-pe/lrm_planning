# Tool: execute_python
# Step: 4


def simulate_moves(initial_state, moves):
    """
    Simulate the moves and verify the result.
    
    Args:
        initial_state: Initial peg configuration
        moves: List of moves [disk_id, from_peg, to_peg]
    
    Returns:
        Final state after all moves
    """
    # Create a copy of the initial state
    state = [peg.copy() for peg in initial_state]
    
    print("Initial state:")
    for i, peg in enumerate(state):
        print(f"Peg {i}: {peg}")
    print()
    
    for i, move in enumerate(moves, 1):
        disk_id, from_peg, to_peg = move
        
        # Validate the move
        if not state[from_peg]:
            raise ValueError(f"Move {i}: Peg {from_peg} is empty!")
        
        if state[from_peg][-1] != disk_id:
            raise ValueError(f"Move {i}: Disk {disk_id} is not on top of peg {from_peg}!")
        
        if state[to_peg] and state[to_peg][-1] < disk_id:
            raise ValueError(f"Move {i}: Cannot place disk {disk_id} on top of smaller disk {state[to_peg][-1]}!")
        
        # Execute the move
        state[from_peg].pop()
        state[to_peg].append(disk_id)
        
        print(f"Move {i}: Move disk {disk_id} from peg {from_peg} to peg {to_peg}")
        print("State:")
        for j, peg in enumerate(state):
            print(f"  Peg {j}: {peg}")
        print()
    
    return state

# Initial configuration
initial_state = [[3, 2, 1], [], []]

# The solution moves
moves = [[1, 0, 2], [2, 0, 1], [1, 2, 1], [3, 0, 2], [1, 1, 0], [2, 1, 2], [1, 0, 2]]

# Simulate the moves
final_state = simulate_moves(initial_state, moves)

# Verify the result
expected_state = [[], [], [3, 2, 1]]
if final_state == expected_state:
    print("✓ SUCCESS! The moves correctly transform the initial configuration to the goal configuration.")
else:
    print("✗ FAILURE! The final state does not match the goal configuration.")
    print("Expected:", expected_state)
    print("Actual:", final_state)
