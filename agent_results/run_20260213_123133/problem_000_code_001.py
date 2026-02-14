# Tool: execute_python
# Step: 2

from collections import deque

# Define initial and goal states
initial_state = ((2,), (3, 1), ())  # Peg0: [2], Peg1: [3,1], Peg2: empty
goal_state = ((), (2,), (3, 1))     # Peg0: empty, Peg1: [2], Peg2: [3,1]

# BFS setup
visited = set()
queue = deque()
queue.append((initial_state, []))  # (state, path)
visited.add(initial_state)

while queue:
    state, path = queue.popleft()
    
    # Check if we reached the goal
    if state == goal_state:
        print("Solution found!")
        print(f"moves = {path}")
        break
    
    # Generate all valid moves from current state
    for from_peg in range(3):
        if not state[from_peg]:
            continue  # No disks to move from this peg
            
        disk = state[from_peg][-1]  # Top disk
        
        for to_peg in range(3):
            if from_peg == to_peg:
                continue  # Can't move to same peg
                
            # Check move validity
            if state[to_peg]:
                target_top = state[to_peg][-1]
                if disk > target_top:
                    continue  # Invalid: larger disk on smaller
                    
            # Create new state
            new_pegs = [list(p) for p in state]
            moved_disk = new_pegs[from_peg].pop()
            new_pegs[to_peg].append(moved_disk)
            new_state = tuple(tuple(p) for p in new_pegs)
            
            # Skip visited states
            if new_state in visited:
                continue
                
            visited.add(new_state)
            new_path = path + [[disk, from_peg, to_peg]]
            queue.append((new_state, new_path))
else:
    print("No solution found!")