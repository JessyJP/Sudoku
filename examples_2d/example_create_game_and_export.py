# Import necessary modules from the sudoku package
from sudoku import *
from sudoku.make_game import select_board_file, make_game_from_solution

# Load a Sudoku board from a specified directory
B = load_sudoku_board(select_board_file("../Solutions/", index=None))
print_board_state(B, substituteZero=".", border=True, clear=True)

# Initialize variables to keep track of the board with the least number of clues
B_least = B.copy()

# Attempt to generate a game with fewer clues multiple times to find the best one
for t in range(300):  # Try 300 times

    # Choose between the 2 options.
    B_current = make_game_from_solution(B.copy())
    # B_current = make_game_from_solution_sequentially(B.copy())

    currentSum = count_non_zero_elements(B_current)
    newSum = count_non_zero_elements(B_least)
    if currentSum < newSum:
        print(f"TEST {t}: Current number count {currentSum} < Smallest number count {newSum}")
        B_least = B_current

# Display the board with the least clues found
print_board_state(B_least, substituteZero=".", border=True, clear=False)

# Save the board to different formats
directory = "r:/"
filename = "sudoku"
fileExt = ".tex"

# Export and compile the board to a LaTeX document
export_board_to_TEX(directory + filename + fileExt, B_least)
compilePDF(directory, filename + fileExt)

# Export the board to an HTML file and open it in a web browser
html_file = directory + filename + ".html"
export_board_to_html(B_least, html_file)
open_html(html_file)

# Open the generated PDF document
open_pdf(directory + filename + ".pdf")
