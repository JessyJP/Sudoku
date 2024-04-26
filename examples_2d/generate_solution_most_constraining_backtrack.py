from sudoku import *  # Consider specifying what you need if * imports are too broad

# =======================================================
# Script Configuration
# =======================================================
# Define grid size for the Sudoku puzzle
g = 5  # Grid size for blocks (e.g., 5 for a 25x25 board)
N = g**2  # Total size of the board (N x N)

# =======================================================
# Initialize Sudoku Board
# =======================================================
# Create an initial empty board
B = InitializeBoard(N)

# Initialize the solution space for the board
S = initialize_solution_space(B)

# Display the initial state of the board
print_board_state(B, border=True)

# =======================================================
# Solve the Sudoku Using Backtracking
# =======================================================
# Attempt to solve the Sudoku using the most constrained backtracking method
B, Done, trialCount_BTG = BacktrackMostConstrained(B, S, refreshCount=1, countLimit=10**6)

# Display the board after attempting to solve
print_board_state(B, substituteZero='.', border=True)
print(f"Trial count: {trialCount_BTG}")

# =======================================================
# Validation and Saving
# =======================================================
# Check if the Sudoku is valid and fully filled
if isValidSudoku(B) and count_non_zero_elements(B) == N**2:
    print("NEW SUDOKU board created successfully!")
    save_sudoku_board(B)  # Save the board if it is successfully created
else:
    print("ERROR: An issue occurred while creating the board!")

