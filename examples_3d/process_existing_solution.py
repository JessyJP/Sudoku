from sudoku import compilePDF, open_html, open_pdf
from sudoku.visualize import update_voxel_colors
from sudoku3d.export_3d import load_sudoku_board_3d, export_board_to_TEX_3d, export_3d_board_to_TEX, \
    export_board_to_html_3d
from sudoku3d.solver_3d import initialize_3d_solution_space
from sudoku3d.visualize_3d import plot_board_state_3d, print_full_solution_space

solution_name_3d_g2 = f"Sudoku3DSolution_9x9x9_{'1'}.csv"

B = load_sudoku_board_3d( "../Solutions/"+solution_name_3d_g2)

# Initialize the solution space for the board
S = initialize_3d_solution_space(B)

# =======================================================
# Exports
# =======================================================
voxel_grid = plot_board_state_3d(B, border=False, substituteZero=".")
update_voxel_colors(voxel_grid, B)

print_full_solution_space(S, border=True)  # Assuming there's a function to display 3D boards

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
