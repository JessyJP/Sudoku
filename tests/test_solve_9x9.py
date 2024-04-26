from sudoku import *
import numpy as np

from sudoku.indexing import convert_to_int_board

## =========================================================================
# Utility Functions
# =========================================================================



## =========================================================================
# Example Run
# =========================================================================

# Initial test board with empty spaces represented as "."
test_board = [
    ["5","3",".",".","7",".",".",".","."],
    ["6",".",".","1","9","5",".",".","."],
    [".","9","8",".",".",".",".","6","."],
    ["8",".",".",".","6",".",".",".","3"],
    ["4",".",".","8",".","3",".",".","1"],
    ["7",".",".",".","2",".",".",".","6"],
    [".","6",".",".",".",".","2","8","."],
    [".",".",".","4","1","9",".",".","5"],
    [".",".",".",".","8",".",".","7","9"]
]

# Board solved for validation
solved_sudoku_board = [
    ["5","3","4","6","7","8","9","1","2"],
    ["6","7","2","1","9","5","3","4","8"],
    ["1","9","8","3","4","2","5","6","7"],
    ["8","5","9","7","6","1","4","2","3"],
    ["4","2","6","8","5","3","7","9","1"],
    ["7","1","3","9","2","4","8","5","6"],
    ["9","6","1","5","3","7","2","8","4"],
    ["2","8","7","4","1","9","6","3","5"],
    ["3","4","5","2","8","6","1","7","9"]
]

# Convert string boards to integer boards
B = convert_to_int_board(test_board)
solved_board = convert_to_int_board(solved_sudoku_board)

# Display initial board state
print_board_state(B, '.', True)
print(f"Filled cells percentage: {np.count_nonzero(B)/(len(B)**2) * 100:.2f}%")

# Initialize solution space and solve
S = initialize_solution_space(B, dispSolutionState=False)
B = solve(B, S, dispBoard=True)

# Display final board state and check against the solved board
print_board_state(B, '.', True)
compare_boards(solved_board, B)
