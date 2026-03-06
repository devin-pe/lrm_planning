# Tool: execute_python
# Step: 4


def simulate_hanoi(moves, n):
    """
    Simulate the Tower of Hanoi moves to verify correctness.
    
    Args:
        moves: List of moves in format [[disk_id, from_peg, to_peg], ...]
        n: Number of disks
    
    Returns:
        Final state of pegs
    """
    # Initialize pegs: peg 0 has all disks (largest at bottom), others empty
    pegs = [list(range(n, 0, -1)), [], []]
    
    print("Initial state:")
    for i, peg in enumerate(pegs):
        print(f"Peg {i}: {peg}")
    print()
    
    # Simulate each move
    for i, move in enumerate(moves, 1):
        disk_id, from_peg, to_peg = move
        
        # Validate move
        if not pegs[from_peg]:
            raise ValueError(f"Move {i}: Peg {from_peg} is empty!")
        
        top_disk = pegs[from_peg][-1]
        if top_disk != disk_id:
            raise ValueError(f"Move {i}: Trying to move disk {disk_id} but disk {top_disk} is on top!")
        
        if pegs[to_peg] and pegs[to_peg][-1] < disk_id:
            raise ValueError(f"Move {i}: Cannot place disk {disk_id} on top of smaller disk {pegs[to_peg][-1]}!")
        
        # Execute move
        pegs[from_peg].pop()
        pegs[to_peg].append(disk_id)
        
        # Print state after each move for first few and last few moves
        if i <= 5 or i >= len(moves) - 5:
            print(f"After move {i:2d} (disk {disk_id}: {from_peg} -> {to_peg}):")
            for j, peg in enumerate(pegs):
                print(f"  Peg {j}: {peg}")
            print()
    
    return pegs

# Simulate the solution
final_state = simulate_hanoi(moves, 5)

print("Final state:")
for i, peg in enumerate(final_state):
    print(f"Peg {i}: {peg}")

# Verify goal achieved
if final_state[0] == [] and final_state[1] == [] and final_state[2] == [5, 4, 3, 2, 1]:
    print("\n✓ SUCCESS: Goal configuration achieved!")
else:
    print("\n✗ FAILURE: Goal configuration not achieved!")