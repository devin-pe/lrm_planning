# Tool: execute_python
# Step: 2

def tower_of_hanoi(n, source, destination, auxiliary, moves):
    if n == 1:
        moves.append([1, source, destination])
        return
    
    tower_of_hanoi(n-1, source, auxiliary, destination, moves)
    moves.append([n, source, destination])
    tower_of_hanoi(n-1, auxiliary, destination, source, moves)

def main():
    n = 4  
    moves = []
    tower_of_hanoi(n, 0, 2, 1, moves)
    print("moves =", moves)

if __name__ == "__main__":
    main()