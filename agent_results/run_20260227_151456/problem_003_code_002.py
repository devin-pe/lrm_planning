# Tool: execute_python
# Step: 4

def simulate_moves(initial_state, moves):
    """
    Simulate the Tower of Hanoi moves and verify the result.
    """
    # Create a copy of the initial state
    pegs = [list(peg) for peg in initial_state]
    
    print("Initial state:")
    for i, peg in enumerate(pegs):
        print(f"Peg {i}: {peg}")
    print()
    
    # Simulate each move
    for i, move in enumerate(moves):
        disk_id, from_peg, to_peg = move
        
        # Validate the move
        if not pegs[from_peg]:
            print(f"ERROR: Move {i+1}: Cannot move disk {disk_id} from empty peg {from_peg}")
            return False
        
        top_disk = pegs[from_peg][-1]
        if top_disk != disk_id:
            print(f"ERROR: Move {i+1}: Disk {disk_id} is not on top of peg {from_peg} (top is {top_disk})")
            return False
        
        # Check if destination peg is empty or has larger disk
        if pegs[to_peg] and pegs[to_peg][-1] < disk_id:
            print(f"ERROR: Move {i+1}: Cannot place disk {disk_id} on top of smaller disk {pegs[to_peg][-1]}")
            return False
        
        # Make the move
        pegs[from_peg].pop()
        pegs[to_peg].append(disk_id)
        
        print(f"Move {i+1}: Disk {disk_id} from peg {from_peg} to peg {to_peg}")
        for j, peg in enumerate(pegs):
            print(f"  Peg {j}: {peg}")
        print()
    
    print("Final state:")
    for i, peg in enumerate(pegs):
        print(f"Peg {i}: {peg}")
    
    return pegs

# Initial configuration for 4 disks
initial_state = [[4, 3, 2, 1], [], []]
final_state = simulate_moves(initial_state, moves)

# Check if we reached the goal
if final_state[0] == [] and final_state[1] == [] and final_state[2] == [4, 3, 2, 1]:
    print("\n✓ SUCCESS: Goal configuration reached!")
else:
    print("\n✗ FAILURE: Did not reach goal configuration")
