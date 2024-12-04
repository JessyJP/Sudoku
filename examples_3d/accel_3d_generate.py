import numpy as np

from sudoku import compilePDF, open_html, open_pdf
from sudoku3d.accel_solver_3d import Initialize3DBoard, initialize_3d_solution_space, backTrackGenerate3D3, \
    isValidSudoku3D_slices
from sudoku3d.export_3d import save_sudoku_board_3d, export_board_to_TEX_3d, export_board_to_html_3d
from sudoku3d.visualize_3d import print_layers, plot_board_state_3d, print_full_solution_space, \
    update_plot_board_state_3d

# =======================================================
# Script Configuration
# =======================================================
# Define grid size for the Sudoku puzzle
g = 3  # Grid size for blocks (e.g., 3 for a 27x27x27 board)
N = g**2  # Total size of the board (N x N x N)

# =======================================================
# Initialize Sudoku Board
# =======================================================
# Create an initial empty board
B = Initialize3DBoard(N)

# Initialize the solution space for the board
S = initialize_3d_solution_space(B)


print_layers(B)
voxel_grid = plot_board_state_3d(B, border=False,substituteZero=".")
# Display the initial state of the board
print_full_solution_space(S)  # Assuming there's a function to display 3D boards
update_plot_board_state_3d(B, voxel_grid, substituteZero='.', border=True, clear=False)

# =======================================================
# Solve the Sudoku Using Backtracking
# =======================================================
# Attempt to solve the Sudoku using the most constrained backtracking method in 3D
# B, success = backtrackMostConstrained3D(B, voxel_grid=voxel_grid)
# B, success = solve3D_by_all_slices(B, S, dispSolutionState=False)
# B = solve3D_alt1(B, S, voxel_grid, dispSolutionState=False)
# B = backTrackGenerate3D(B, refreshCount=1000, maxTrials=None)
B, success = backTrackGenerate3D3(B, random_try=True, random_order=False)

# Display the board after attempting to solve
print_full_solution_space(S)  # Assuming there's a function to display 3D boards
update_plot_board_state_3d(B, voxel_grid=voxel_grid, substituteZero='.', border=False, clear=False)
if success:
    print("NEW SUDOKU 3D board created successfully!")
else:
    print("ERROR: An issue occurred while creating the 3D board!")

# =======================================================
# Validation and Saving
# =======================================================
print_layers(B)
# Check if the Sudoku is valid and fully filled
if isValidSudoku3D_slices(B) and np.all(B != 0):
    print("The 3D Sudoku board is valid and complete.")
    # Optionally save or further process the generated board
    save_sudoku_board_3d(B)  # Save the board if it is successfully created
    # Save the board to different formats
    directory = "r:/"
    filename = "sudoku"
    fileExt = ".tex"

    # Export and compile the board to a LaTeX document
    export_board_to_TEX_3d(directory + filename + fileExt, B)
    # export_3d_board_to_TEX(directory + filename + fileExt, B)
    compilePDF(directory, filename + fileExt)

    # Export the board to an HTML file and open it in a web browser
    html_file = directory + filename + ".html"
    export_board_to_html_3d(B, html_file)
    # open_html(html_file)

    # Open the generated PDF document
    open_pdf(directory + filename + ".pdf")

else:
    print("The board is either incomplete or invalid.")
