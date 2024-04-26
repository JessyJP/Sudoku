import os
import numpy as np
from random import choice
from .solver import solve, initialize_solution_space
from .visualize import print_board_state


def select_board_file(directory, index=None):
    """
    Select a Sudoku board file from a specified directory based on an optional index. If no index
    is provided, or if the index is out of range, a random file is selected.

    Parameters:
    - directory: The directory containing board files.
    - index: Optional integer to specify which file to select.

    Returns:
    - The path to the selected board file.
    """
    files = os.listdir(directory)  # Get a list of files in the directory

    # If the index is not provided or is out of range, pick a random file
    if index is None or index >= len(files):
        selected_file = choice(files)
    else:
        selected_file = files[index]

    file_path = os.path.join(directory, selected_file)
    print(f"File selected '{selected_file}' in directory: '{directory}'")
    return file_path


def remove_elements(B, n, row_indices, col_indices):
    """
    Helper function to set the first n elements of the board to zero based on shuffled indices.

    Parameters:
    - B: The Sudoku board as a 2D numpy array.
    - n: Number of elements to remove.
    - row_indices: Shuffled array of row indices where elements will be removed.
    - col_indices: Shuffled array of column indices where elements will be removed.

    Returns:
    - The modified board with n elements set to zero.
    """
    for i in range(n):
        B[row_indices[i], col_indices[i]] = 0
    return B


def make_game_from_solution_sequentially(B_full, display=False):
    """
    Generate a playable Sudoku game from a full solution by removing elements one by one,
    ensuring the board remains uniquely solvable at each step.

    Parameters:
    - B_full: A full Sudoku solution as a 2D numpy array.
    - display: Boolean indicating whether to display the board state during processing.

    Returns:
    - A 2D numpy array representing the generated Sudoku game with some elements removed.
    """
    row_indices, col_indices = np.where(B_full > 0)  # Get indices of non-zero elements
    permutation = np.random.permutation(len(row_indices))
    Rs = row_indices[permutation]
    Cs = col_indices[permutation]

    B_test = B_full.copy()
    for i in range(len(Rs)):
        element_to_remove = B_test[Rs[i], Cs[i]]
        B_test[Rs[i], Cs[i]] = 0  # Remove one element at a time

        if display:
            print_board_state(B_test, substituteZero=".", border=True)

        S = initialize_solution_space(B_test)
        try:
            solved_board = solve(B_test.copy(), S, dispSolutionState=False)
            if not np.array_equal(solved_board, B_full):
                B_test[Rs[i], Cs[i]] = element_to_remove
                break
        except Exception as e:
            # Reinsert the last removed element
            B_test[Rs[i], Cs[i]] = element_to_remove
            solved_board = solve(B_test.copy(), S, dispSolutionState=display)
            break

    if display:
        print_board_state(B_test, substituteZero=".", border=True)
        print(f"Current number count {np.count_nonzero(B_test)}")

    return B_test


def make_game_from_solution(B_full, display=False):
    """
    Generate a playable Sudoku game from a full solution by strategically removing elements to
    create a puzzle.

    Parameters:
    - B_full: A full Sudoku solution as a 2D numpy array.
    - display: Boolean indicating whether to display the board state during processing.

    Returns:
    - A 2D numpy array representing the generated Sudoku game with some elements removed.
    """
    size = B_full.size

    # Get the indices of the elements in the matrix
    row_indices, col_indices = np.where(B_full > 0)  # Ensure we're only considering non-zero entries

    # Generate a permutation of the indices
    permutation = np.random.permutation(len(row_indices))

    # Shuffle the row and column indices according to permutation
    Rs = row_indices[permutation]
    Cs = col_indices[permutation]

    upper = size
    lower = 0
    while upper != lower and abs(upper - lower) > 1:
        n = int((upper + lower) / 2)
        B_test = remove_elements(B_full.copy(), n, Rs, Cs)

        if display:
            print_board_state(B_test, substituteZero=".", border=True)
            print(f"\nTest Remove n: {n} with DEL Upper Limit: {upper} & DEL Lower Limit: {lower}")

        S = initialize_solution_space(B_test)

        try:
            B_test = solve(B_test, S, dispSolutionState=display)
        except Exception as e:
            # print(f"Error during solving: {str(e)}")
            upper = n  # Adjust upper limit if solving fails
            continue

        if np.array_equal(B_test, B_full):
            lower = n
        else:
            upper = n

    B_game = remove_elements(B_full.copy(), upper, Rs, Cs)

    if display:
        print_board_state(B_game, substituteZero=".", border=True)
        print(f"Current number count {np.count_nonzero(B_game)}")
    return B_game
