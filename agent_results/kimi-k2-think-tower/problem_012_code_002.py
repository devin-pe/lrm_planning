# Tool: execute_python
# Step: 4


def simulate_hanoi(moves, n):
    """
    Simulate the Tower of Hanoi moves to verify correctness.
    
    Args:
        moves: List of moves in format [disk_id, from_peg, to_peg]
        n: Number of disks
    
    Returns:
        True if moves are valid and reach goal, False otherwise
    """
    # Initialize pegs: peg 0 has disks n to 1 (largest to smallest)
    pegs = [list(range(n, 0, -1)), [], []]
    
    print("Initial state:")
    for i, peg in enumerate(pegs):
        print(f"Peg {i}: {peg}")
    print()
    
    for i, move in enumerate(moves, 1):
        disk_id, from_peg, to_peg = move
        
        # Check if move is valid
        if not pegs[from_peg]:
            print(f"Error at move {i}: Peg {from_peg} is empty!")
            return False
        
        # Check if the top disk matches the disk_id
        if pegs[from_peg][-1] != disk_id:
            print(f"Error at move {i}: Top disk on peg {from_peg} is {pegs[from_peg][-1]}, not {disk_id}!")
            return False
        
        # Check if we can place disk on target peg (no larger disk on smaller disk)
        if pegs[to_peg] and pegs[to_peg][-1] < disk_id:
            print(f"Error at move {i}: Cannot place disk {disk_id} on top of smaller disk {pegs[to_peg][-1]}!")
            return False
        
        # Make the move
        disk = pegs[from_peg].pop()
        pegs[to_peg].append(disk)
        
        print(f"Move {i:2d}: Disk {disk_id} from peg {from_peg} to peg {to_peg}")
        print(f"         State: Peg 0: {pegs[0]}, Peg 1: {pegs[1]}, Peg 2: {pegs[2]}")
    
    # Check if we reached the goal
    if pegs[0] == [] and pegs[1] == [] and pegs[2] == list(range(n, 0, -1)):
        print(f"\nSuccess! Reached goal configuration in {len(moves)} moves.")
        return True
    else:
        print(f"\nFailed! Final state is not the goal configuration.")
        print(f"Final state: Peg 0: {pegs[0]}, Peg 1: {pegs[1]}, Peg 2: {pegs[2]}")
        return False

# Verify the moves
moves = [[1, 0, 1], [2, 0, 2], [1, 1, 2], [3, 0, 1], [1, 2, 0], [2, 2, 1], [1, 0, 1], [4, 0, 2], [1, 1, 2], [2, 1, 0], [1, 2, 0], [3, 1, 2], [1, 0, 1], [2, 0, 2], [1, 1, 2]]
success = simulate_hanoi(moves, 4)
