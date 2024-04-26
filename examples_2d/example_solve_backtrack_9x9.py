# Import necessary modules and functions from the sudoku package
from sudoku import convert_to_int_board, print_board_state, compare_boards, backTrackSolve
import os

# Sample Sudoku board to be solved
testBoard = [
    ["5", "3", ".", ".", "7", ".", ".", ".", "."],
    ["6", ".", ".", "1", "9", "5", ".", ".", "."],
    [".", "9", "8", ".", ".", ".", ".", "6", "."],
    ["8", ".", ".", ".", "6", ".", ".", ".", "3"],
    ["4", ".", ".", "8", ".", "3", ".", ".", "1"],
    ["7", ".", ".", ".", "2", ".", ".", ".", "6"],
    [".", "6", ".", ".", ".", ".", "2", "8", "."],
    [".", ".", ".", "4", "1", "9", ".", ".", "5"],
    [".", ".", ".", ".", "8", ".", ".", "7", "9"]
]

# Correctly solved Sudoku board for validation
solved_sudoku_board = [
    ["5", "3", "4", "6", "7", "8", "9", "1", "2"],
    ["6", "7", "2", "1", "9", "5", "3", "4", "8"],
    ["1", "9", "8", "3", "4", "2", "5", "6", "7"],
    ["8", "5", "9", "7", "6", "1", "4", "2", "3"],
    ["4", "2", "6", "8", "5", "3", "7", "9", "1"],
    ["7", "1", "3", "9", "2", "4", "8", "5", "6"],
    ["9", "6", "1", "5", "3", "7", "2", "8", "4"],
    ["2", "8", "7", "4", "1", "9", "6", "3", "5"],
    ["3", "4", "5", "2", "8", "6", "1", "7", "9"]
]

# Convert string boards to integer boards for processing
testBoard = convert_to_int_board(testBoard)
solved_sudoku_board = convert_to_int_board(solved_sudoku_board)

# Clear the console before printing new output
os.system('cls' if os.name == 'nt' else 'clear')

# Solve the Sudoku using backtracking and display the results
(testBoard, trialCount) = backTrackSolve(testBoard, refreshCount=100)
print_board_state(testBoard, substituteZero='.', border=True)
print(f"Trial count: {trialCount}")

# Compare the solved board with the expected solution
compare_boards(testBoard, solved_sudoku_board)
