# Tool: execute_python
# Step: 2

def tower_of_hanoi(n, source, target, auxiliary, moves):
    if n > 0:
        # Move n-1 disks from source to auxiliary
        tower_of_hanoi(n-1, source, auxiliary, target, moves)
        
        # Move the nth disk from source to target
        moves.append([n, source, target])
        
        # Move n-1 disks from auxiliary to target
        tower_of_hanoi(n-1, auxiliary, target, source, moves)

def main():
    n = 3  # Number of disks
    moves = []
    tower_of_hanoi(n, 0, 2, 1, moves)
    print('moves =', moves)

if __name__ == "__main__":
    main()