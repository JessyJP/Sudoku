from sudoku.solver import *
import numpy as np

# =======================================================
# Sudoku Board Initialization
# =======================================================
# Set grid size for the blocks
g = 3
N = g**2  # Calculate total size of the board (N x N)

# Initialize an empty board and solution space
B = InitializeBoard(N)
S = initialize_solution_space(B)

# Solve the board using the most constrained backtracking method
(B, Done, trialCount_BTG) = BacktrackMostConstrained(B, S, refreshCount=1)

# =======================================================
# Utility Functions for Sudoku Expansion
# =======================================================
def add_one_after_g(vec, g):
    """
    Adds 1 to elements in the vector after every 'g' elements to accommodate expanded grid indices.

    Parameters:
    - vec: The original array of indices.
    - g: The block size in the original Sudoku grid.

    Returns:
    - Modified vector with incremented indices after every 'g' elements.
    """
    n_times = len(vec) // g
    for i in range(1, n_times + 1):
        vec[i * g:] += 1
    return vec

def expand_sudoku(B):
    """
    Expands a Sudoku board from size N to (N+1) assuming N is a perfect square.

    Parameters:
    - B: The original NxN Sudoku board.

    Returns:
    - An expanded (N+1)x(N+1) Sudoku board with the original values mapped accordingly.
    """
    N = len(B)
    g = int(np.sqrt(N))
    r = np.array(range(g**2))
    r = add_one_after_g(r, g)

    B_new = InitializeBoard((g + 1) ** 2)
    # Copy the old matrix values into the new one adjusting for expanded indices
    for i in range(B.shape[0]):
        for j in range(B.shape[1]):
            B_new[r[i], r[j]] = B[i, j]
            print_board_state(B_new, substituteZero='.', border=True)
    return B_new

# =======================================================
# Expanding and Solving the Expanded Sudoku
# =======================================================
# Expand the Sudoku board
B = expand_sudoku(B)

# Reinitialize solution space for expanded board and solve
S = initialize_solution_space(B)
B = solve(B, S, dispBoard=True)

# Perform backtracking to solve the Sudoku
(B, Done, trialCount_BTG) = BacktrackMostConstrained(B, S, refreshCount=1)

# Generate a new Sudoku game using a more relaxed backtracking with a higher refresh count
B = backTrackGenerate(B, refreshCount=1000)

