from sudoku import *

# =======================================================
# Section 1: Initialization
# =======================================================
# Initialize a Sudoku board with a predefined grid size
g = 4  # size of the blocks (e.g., 4 for a 16x16 board)
N = g**2
B = InitializeBoard(N)  # Initialize an empty N x N board
S = initialize_solution_space(B)  # Initialize the solution space for the board

# =======================================================
# Section 2: Index Order Generation
# =======================================================
# Generate various index orders to guide the puzzle generation algorithm
RC_zigzag = get_zigzag_matrix_indices(N)  # Get indices for zigzag traversal
RC_diag_block1 = get_diagonal_block_indices(g)  # Get indices for diagonal blocks
RC_diag_block2 = get_diagonal_block_indices(g, reverse=True)  # Get indices for reverse diagonal blocks
R_linear = get_linear_coordinates(N)  # Get linear coordinates for the board
RC = merge_unique_ordered(RC_diag_block1, RC_diag_block2)  # Merge two sets of coordinates

RC_zigzag = get_zigzag_matrix_indices(N)
# Optionally merge more coordinates (currently commented out)
# RC = merge_unique_ordered(RC, R_linear)

# Convert coordinate tuples to separate vectors
elementCorrdinateVectors = coordinates_to_vectors(RC)

# =======================================================
# Section 3: Board Generation
# =======================================================
# Generate a new Sudoku board using backtracking with custom index vectors
(B, trialCount) = backTrackGenerate(B, refreshCount=1, indexVectors=elementCorrdinateVectors)

# Re-initialize the solution space and solve the board
S = initialize_solution_space(B)
B = solve(B, S, dispBoard=True)

# Merge additional coordinates and regenerate the board
RC = merge_unique_ordered(RC, R_linear)
elementCorrdinateVectors = coordinates_to_vectors(RC)
(B, trialCount) = backTrackGenerate(B, refreshCount=100, indexVectors=elementCorrdinateVectors)

# Display the final board state and trial count
print_board_state(B, '.', True)
print(f"Trial count: {trialCount}")

# =======================================================
# Section 4: Validation and Saving
# =======================================================
# Validate the created Sudoku board and save if valid
if isValidSudoku(B) and count_non_zero_elements(B) == N**2:
    print("NEW SUDOKU board created successfully!")
    save_sudoku_board(B)  # Save the board to a file
else:
    print("ERROR: An issue occurred while creating the board!")

# # =======================================================
# # Section 5: Export and Open Functions
# # =======================================================
# # Define file paths and extensions for exports
# directory = "r:/"
# filename = "sudoku"
# fileExt = ".tex"
#
# # Export and compile the board to a LaTeX document
# export_board_to_TEX(directory + filename + fileExt, B)
# compilePDF(directory, filename + fileExt)
#
# # Export the board to an HTML file and open it in a web browser
# html_file = directory + filename + ".html"
# export_board_to_html(B, html_file)
# open_html(html_file)
#
# # Open the generated PDF document
# open_pdf(directory + filename + ".pdf")
