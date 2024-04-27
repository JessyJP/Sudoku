import time
import numpy as np
import random

from sudoku import compare_boards
from sudoku.solver import isValidSudoku, solve
from .indexing_3d import get_random_element_indices_of_non_zero_3d
from .visualize_3d import print_full_solution_space


# ==========================================================
# Sudoku Library: Utilities for handling Sudoku puzzles
# ==========================================================


def Initialize3DBoard(N):
    """
    Initialize a 3D board for Sudoku with all cells set to zero.

    Args:
    - N (int): The dimension of the Sudoku board (N x N x N).

    Returns:
    - numpy.ndarray: A 3D array (N x N x N) filled with zeros.
    """
    return np.zeros((N, N, N), dtype=int)


def initialize_3d_solution_space(B, dispSolutionState=False):
    """
    Initialize the solution space for 3D Sudoku with possible values for each cell.

    Args:
    - B (numpy.ndarray): The 3D Sudoku board (N x N x N).
    - dispSolutionState (bool): Whether to display the solution space updates.

    Returns:
    - numpy.ndarray: A 4D array representing the solution space for each cell.
    """
    N = len(B)
    # Initialize all values as possible
    S = np.full((N, N, N, N), True)

    # Iterate through each layer, row, and column

    for r in range(N):
        for c in range(N):
            for z in range(N):
                if B[r, c, z] > 0:  # If the cell is already filled
                    S = cubeOp(S, B, r, c, z, dispSolutionState)
    return S


def cubeOp(S, B, r, c, z, dispSolutionState=False):
    """
    Apply Sudoku rules in a 3D cube to eliminate impossible values from the solution space.

    Args:
    - S (numpy.ndarray): The 4D solution space array for the Sudoku puzzle.
    - B (numpy.ndarray): The 3D Sudoku board.
    - r, c, z (int): Row, column, and depth indices of the modified cell in the board.
    - dispSolutionState (bool): If True, display the solution state changes.

    Returns:
    - numpy.ndarray: Updated solution space.
    """
    N = B.shape[0]  # Assuming a cubic NxNxN board.
    val = B[r, c, z] - 1  # Subtract 1 to convert the cell's value to a zero-indexed position.

    # Invalidate all possibilities in the (r, c, z) cell itself
    S[r, c, z, :] = False

    # Invalidate in the same row across all layers
    S[r, c, :, val] = False
    # Invalidate in the same column across all columns and
    S[r, :, z, val] = False
    # Invalidate in the same layer across all rows
    S[:, c, z, val] = False

    # Invalidate all other cells in the same block (sub-grid)
    block_size = int(np.sqrt(N))  # Assuming cubic blocks in a cubic grid
    R0 = (r // block_size) * block_size
    C0 = (c // block_size) * block_size
    Z0 = (z // block_size) * block_size

    # Check the r-c plane in fixed z
    S[R0:R0 + block_size, C0 : C0 + block_size, z, val] = False

    # Check the r-z plane in fixed c
    S[R0:R0 + block_size, c, Z0:Z0 + block_size, val] = False

    # Check the c-z plane in fixed r
    S[r, C0 : C0 + block_size,  Z0:Z0 + block_size, val] = False

    if dispSolutionState:
        print_full_solution_space(B)
        print("Updated solution space at position ({}, {}, {}) for value {}".format(r, c, z, val + 1))

    return S

def isValidSudoku3D_other(B):
    """
    Check if a 3D Sudoku board is valid.

    Args:
    - B (numpy.ndarray): The 3D Sudoku board (N x N x N).

    Returns:
    - bool: True if the board is valid, False otherwise.
    """
    N = B.shape[0]  # Assuming a cubic NxNxN board
    block_size = int(np.sqrt(N))  # Assuming cubic blocks

    if block_size**3 != N:
        print("The block size or board dimensions are invalid for a 3D Sudoku.")
        return False

    # Checking each number once per row, column, and layer
    seen = set()

    # Check rows, columns, and layers
    for i in range(N):
        for j in range(N):
            row_set = set()
            col_set = set()
            layer_set = set()

            for k in range(N):
                # Check rows
                if B[i][j][k] in row_set:
                    return False
                elif B[i][j][k] != 0:
                    row_set.add(B[i][j][k])

                # Check columns
                if B[i][k][j] in col_set:
                    return False
                elif B[i][k][j] != 0:
                    col_set.add(B[i][k][j])

                # Check layers
                if B[k][i][j] in layer_set:
                    return False
                elif B[k][i][j] != 0:
                    layer_set.add(B[k][i][j])

    # Check blocks
    for x in range(0, N, block_size):
        for y in range(0, N, block_size):
            for z in range(0, N, block_size):
                block_set = set()
                for dx in range(block_size):
                    for dy in range(block_size):
                        for dz in range(block_size):
                            val = B[x + dx][y + dy][z + dz]
                            if val in block_set:
                                return False
                            elif val != 0:
                                block_set.add(val)

    return True


def isValidSudoku3D_slices(board):
    """Check if a 3D Sudoku board is valid across all slices in three dimensions."""
    N = len(board)  # Assuming the board is N x N x N
    # Validate each horizontal layer
    for z in range(N):
        if not isValidSudoku(board[:, :, z]):
            return False

    # Validate each vertical slice along x-axis (column-wise slices for each depth)
    for x in range(N):
        slice_x = np.squeeze(np.take(board, x, axis=0))
        if not isValidSudoku(slice_x):
            return False

    # Validate each vertical slice along y-axis (row-wise slices for each depth)
    for y in range(N):
        slice_y = np.squeeze(np.take(board, y, axis=1))
        if not isValidSudoku(slice_y):
            return False

    return True


def isValidPlacement3D_block(B, x, y, z, num):
    """
    Check if it's safe to place a number in the specified cell in a 3D Sudoku grid.

    Args:
    - B (numpy.ndarray): The 3D Sudoku board.
    - x, y, z (int): Indices in the board.
    - num (int): Number to place.

    Returns:
    - bool: True if safe, False otherwise.
    """
    N = B.shape[0]
    block_size = int(np.sqrt(N))  # Assuming cubic blocks

    # Check line constraints
    if num in B[x, y, :] or num in B[x, :, z] or num in B[:, y, z]:
        return False

    # Calculate block start indices
    block_start_x = (x // block_size) * block_size
    block_start_y = (y // block_size) * block_size
    block_start_z = (z // block_size) * block_size

    # Check within the block
    for i in range(block_start_x, block_start_x + block_size):
        for j in range(block_start_y, block_start_y + block_size):
            for k in range(block_start_z, block_start_z + block_size):
                if B[i, j, k] == num:
                    return False
    return True

# =========================================================================
# Normal Solve with elimination
# =========================================================================

def solve3D_by_all_slices(B, S, dispBoard=False, dispSolutionState=False):
    N = B.shape[0]  # Assuming the board is N x N x N

    def solve_slice(slice_2d, solution_space_2d):
        """Apply the 2D solve function to a 2D slice."""
        # Assuming 'solve' is a callable that works for 2D boards
        slice_2d = solve(slice_2d, solution_space_2d, dispBoard, dispSolutionState)
        return (slice_2d,solution_space_2d)

    # Iterate over each horizontal layer
    for z in range(N):
        B[:, :, z], S[:, :, z, :] = solve_slice(B[:, :, z], S[:, :, z, :])

    # Iterate over each vertical slice along x-axis (column-wise slices for each depth)
    for x in range(N):
        slice_x = np.squeeze(B[x, :, :])
        solution_space_x = np.squeeze(S[x, :, :, :])
        solved_slice_x, _ = solve_slice(slice_x, solution_space_x)
        B[x, :, :] = solved_slice_x.reshape((N, N))

    # Iterate over each vertical slice along y-axis (row-wise slices for each depth)
    for y in range(N):
        slice_y = np.squeeze(B[:, y, :])
        solution_space_y = np.squeeze(S[:, y, :, :])
        solved_slice_y, _ = solve_slice(slice_y, solution_space_y)
        B[:, y, :] = solved_slice_y.reshape((N, N))

    # Optionally, re-check validity after attempting to solve all slices
    if not isValidSudoku3D_slices(B):
        print("The board is not solvable or an error occurred during solving.")
        return B, False

    return B, True


def solve3D_alt1(B, S, voxel_grid, dispBoard=False, dispSolutionState=False):
    N = len(B)
    g = int(np.sqrt(N))  # Assuming cubic blocks for simplicity
    keepChecking = True

    def setValueOnBoard(B, S, x, y, z, val):
        nonlocal keepChecking
        # Set value on the board
        if B[x][y][z] == 0:
            B[x][y][z] = val + 1  # Adjust because values in `S` are zero-indexed
            S = cubeOp(S, B, x, y, z, dispSolutionState)
            keepChecking = True
        if dispBoard:
            voxel_grid[x, y, z].update(value=val)
            # print_full_solution_space(B)  # Assuming a function to print the 3D board
        if dispSolutionState:
            print_full_solution_space(S)
        return B, S

    while keepChecking:
        keepChecking = False
        # Check each block for any single option
        for block_x in range(0, N, g):
            for block_y in range(0, N, g):
                for block_z in range(0, N, g):
                    # Check within each block
                    block_vals = np.sum(S[block_x:block_x+g, block_y:block_y+g, block_z:block_z+g, :], axis=(0, 1, 2))
                    for val in range(N):
                        if block_vals[val] == 1:  # Only one cell in the block can take this value
                            # Find the cell
                            for dx in range(g):
                                for dy in range(g):
                                    for dz in range(g):
                                        if S[block_x+dx][block_y+dy][block_z+dz][val]:
                                            B, S = setValueOnBoard(B, S, block_x+dx, block_y+dy, block_z+dz, val)
                                            break

        # Check each row, column, and depth line
        for i in range(N):
            # Rows and columns in each depth
            for j in range(N):
                row_vals = np.sum(S[i, j, :, :], axis=0)
                col_vals = np.sum(S[i, :, j, :], axis=0)
                depth_vals = np.sum(S[:, i, j, :], axis=0)
                for val in range(N):
                    if row_vals[val] == 1:
                        for z in range(N):
                            if S[i][j][z][val]:
                                B, S = setValueOnBoard(B, S, i, j, z, val)
                                break
                    if col_vals[val] == 1:
                        for y in range(N):
                            if S[i][y][j][val]:
                                B, S = setValueOnBoard(B, S, i, y, j, val)
                                break
                    if depth_vals[val] == 1:
                        for x in range(N):
                            if S[x][i][j][val]:
                                B, S = setValueOnBoard(B, S, x, i, j, val)
                                break

    return B


def solve3D_alt2(B, S, dispBoard=False, dispSolutionState=False):
    N = len(B)
    g = int(np.sqrt(N))  # Assuming cubic blocks for simplicity

    def setValueOnBoard(B, S, x, y, z, val):
        if B[x, y, z] == 0:
            B[x, y, z] = val + 1  # Adjust for 1-based index
            S = cubeOp(S, B, x, y, z, dispSolutionState)
        if dispBoard:
            print_full_solution_space(B)  # Assuming a function to print the 3D board
        if dispSolutionState:
            print_full_solution_space(S)
        return B, S

    def process_single_option(axis=0):
        nonlocal B, S
        for idx in range(N):
            # Depending on axis, we choose the slice differently:
            # axis 0 -> rows, axis 1 -> columns, axis 2 -> depths
            if axis == 0:  # Process by row
                option_counts = np.sum(S[idx, :, :, :], axis=(0, 1))
            elif axis == 1:  # Process by column
                option_counts = np.sum(S[:, idx, :, :], axis=(0, 1))
            else:  # Process by depth
                option_counts = np.sum(S[:, :, idx, :], axis=(0, 1))

            for val in range(N):
                if option_counts[val] == 1:  # Find the unique cell
                    if axis == 0:  # Process by row
                        coordinates = np.argwhere(S[idx, :, :, val] == True)
                    elif axis == 1:  # Process by column
                        coordinates = np.argwhere(S[:, idx, :, val] == True)
                    else:  # Process by depth
                        coordinates = np.argwhere(S[:, :, idx, val] == True)

                    for coord in coordinates:
                        x, y, z = coord if axis == 0 else (coord[1], coord[0], coord[2]) if axis == 1 else (
                        coord[2], coord[0], coord[1])
                        B, S = setValueOnBoard(B, S, x, y, z, val)

    keepChecking = True
    while keepChecking:
        previous_state = np.copy(B)
        # Process each type of slice: rows, columns, and depths
        for axis in range(3):
            process_single_option(axis)

        # Process sub-cubes
        for x in range(0, N, g):
            for y in range(0, N, g):
                for z in range(0, N, g):
                    block_possibilities = np.sum(S[x:x + g, y:y + g, z:z + g, :], axis=(0, 1, 2))
                    for val in range(N):
                        if block_possibilities[val] == 1:
                            # Locate the exact cell this value must be in
                            locs = np.argwhere(S[x:x + g, y:y + g, z:z + g, val])
                            for loc in locs:
                                dx, dy, dz = loc
                                B, S = setValueOnBoard(B, S, x + dx, y + dy, z + dz, val)

        # Determine if another pass is needed
        if np.array_equal(B, previous_state):
            keepChecking = False

        if dispBoard:
            print_full_solution_space(B)  # Assuming a function to print the 3D board
        if dispSolutionState:
            print_full_solution_space(S)

    return B


# =========================================================================
# Backtracking Solve
# =========================================================================

def backTrackSolve3D(B):
    """
    Solve the 3D Sudoku board using a backtracking approach with optimized index traversal.

    Args:
    - B (numpy.ndarray): The 3D Sudoku board (N x N x N).

    Returns:
    - tuple: (board, success flag)
    """
    N = B.shape[0]
    g = int(np.sqrt(N))  # Assuming cubic blocks for simplicity
    if not findUnassignedLocation(B):
        return B, True  # Puzzle solved

    # Enhanced with random index traversal
    X, Y, Z = get_random_element_indices_of_non_zero_3d(B == 0)
    for i in range(len(X)):
        x, y, z = X[i], Y[i], Z[i]
        for num in range(1, N+1):
            if isValidSudoku3D_slices(B, x, y, z, num):
                B[x][y][z] = num  # Try potential number
                if backTrackSolve3D(B)[1]:
                    return B, True  # Return if success
                B[x][y][z] = 0  # Reset on failure

    return B, False  # Trigger backtracking


def findUnassignedLocation(B):
    """
    Enhanced to directly use numpy for finding unassigned locations.
    """
    N = B.shape[0]
    pos = np.argwhere(B == 0)
    if len(pos) > 0:
        return tuple(pos[0])  # return first unassigned location
    return None


# =========================================================================
# Backtracking Solution Generation
# =========================================================================


def backTrackGenerate3D(B, refreshCount=1000, maxTrials=None):
    """
    Generate a 3D Sudoku puzzle ensuring it remains solvable.

    Args:
    - B (numpy.ndarray): Partially filled or empty 3D Sudoku board.
    - refreshCount (int): Frequency of status updates for long-running operations.
    - maxTrials (int): Maximum attempts for trying different numbers in cells.

    Returns:
    - tuple: (board, success flag)
    """
    N = B.shape[0]
    g = int(np.sqrt(N))  # Assuming cubic blocks
    trialCount = 0
    trialLimit = maxTrials if maxTrials is not None else N * N * N * N

    def solve_randomly(x, y, z):
        nonlocal trialCount
        possibilities = list(range(1, N + 1))
        random.shuffle(possibilities)  # Randomize the order of numbers to place

        for num in possibilities:
            if isValidSudoku3D_slices(B):
                B[x, y, z] = num
                trialCount += 1
                if trialCount % refreshCount == 0:
                    print_full_solution_space(B)
                    print(f"Trial count: {trialCount}")

                if trialCount >= trialLimit:
                    return False  # Too many trials, likely no solution from this path

                if backTrackGenerate3D(B)[1]:
                    return True
                B[x, y, z] = 0  # Reset if not leading to a solution

        return False

    # Find the first empty cell
    for x in range(N):
        for y in range(N):
            for z in range(N):
                if B[x, y, z] == 0:
                    if not solve_randomly(x, y, z):
                        return B, False
                    return B, True

    return B, True  # If no empty cell is found, the board is already filled


def backtrackMostConstrained3D(B, voxel_grid=None):
    """
    Solve or generate a 3D Sudoku puzzle using the most constrained cell approach.

    Args:
    - B (numpy.ndarray): The 3D Sudoku board (N x N x N).

    Returns:
    - tuple: (board, success flag)
    """
    N = B.shape[0]
    g = int(np.sqrt(N))  # Assuming cubic blocks

    if np.all(B > 0):  # Check if the board is already completely filled
        return B, True

    # Get the most constrained cell
    x, y, z = findMostConstrainedCell3D(B)

    # Try to place a valid number in the most constrained cell
    for num in range(1, N + 1):
        if isValidSudoku3D_slices(B, x, y, z, num):
            B[x, y, z] = num
            update_plot(B,x,y,z,voxel_grid)
            B, success = backtrackMostConstrained3D(B, voxel_grid=voxel_grid)
            if success:
                return B, True
            B[x, y, z] = 0  # Reset cell
            update_plot(B,x,y,z,voxel_grid)

    return B, False

def findMostConstrainedCell3D(B):
    """
    Find the cell with the fewest possible numbers that can be placed, in a 3D Sudoku grid.

    Args:
    - B (numpy.ndarray): The 3D Sudoku board.

    Returns:
    - tuple: (x, y, z) indices of the most constrained cell.
    """
    N = B.shape[0]
    min_options = N + 1  # More than the maximum number of options possible
    min_cell = None

    for x in range(N):
        for y in range(N):
            for z in range(N):
                if B[x, y, z] == 0:  # Only consider empty cells
                    options_count = sum(isValidSudoku3D_slices(B, x, y, z, num) for num in range(1, N + 1))
                    if options_count < min_options:
                        min_options = options_count
                        min_cell = (x, y, z)
                        if min_options == 1:
                            return min_cell  # Return immediately if there's a cell with only one possible number

    return min_cell




def backTrackGenerate3D2(B, voxel_grid=None, refreshCount=1000, maxTrials=None):
    N = len(B)  # Assuming B is N x N x N
    trialCount = 0
    maxTrials = maxTrials if maxTrials is not None else 10**6  # Set a default limit to prevent infinite loops

    # Helper function to attempt to fill the board starting from the first empty cell found
    def solve(depth=0):
        nonlocal trialCount

        # Find the first empty cell
        try:
            x, y, z = np.argwhere(B == 0)[0]
        except IndexError:  # If no empty cell is found, the board is complete
            return True

        # Randomize the numbers to try for a more diverse generation
        numbers = np.random.permutation(N) + 1  # Numbers are 1 to N
        for number in numbers:
            if isValidSudoku3D_slices(B, x, y, z, number):
                B[x, y, z] = number
                trialCount += 1
                update_plot(B, x, y, z, voxel_grid)
                if trialCount % refreshCount == 0:
                    print(f"Trial count: {trialCount}")
                    print_full_solution_space(B)  # Display the board's current state

                if solve(depth + 1):
                    return True

                # Backtrack
                B[x, y, z] = 0
                update_plot(B, x, y, z, voxel_grid)

            if trialCount >= maxTrials:
                print("Reached maximum trials, stopping...")
                return False

        return False

    if solve():
        return B, True  # Successful generation
    else:
        return B, False  # Failed to generate a valid solution


from sudoku3d.visualize_3d import *
import time

def update_plot(B, x, y, z, voxel_grid, delay=0.01):
    """
    Update the plot with a new value at position (x, y, z) and pause for the specified delay.

    Args:
    - B (numpy.ndarray): The 3D Sudoku board.
    - x (int): The x-coordinate in the board.
    - y (int): The y-coordinate in the board.
    - z (int): The z-coordinate in the board.
    - voxel_grid: A 3D voxel grid representation for visualization.
    - delay (float): Time in seconds to delay the update for visualization.

    Returns:
    - None
    """
    val = B[x, y, z] if B[x, y, z] > 0 else "."
    if voxel_grid is not None:
        voxel_grid[x, y, z].update(value=val)  # Assuming a method .update exists to handle the update
    plt.draw()
    plt.pause(delay)  # This both draws the current figure's changes and waits
    # time.sleep(delay)  # Pause the script to allow visualization of changes




import random
def backTrackGenerate3D3(B, voxel_grid=None, refreshCount=1000, random_order=False, random_try=False):
    N = len(B)  # Assuming B is N x N x N
    possible_set = list(range(1, N + 1))
    # Flatten the 3D indices to a list of tuples for easier traversal
    indices = [(x, y, z) for x in range(N) for y in range(N) for z in range(N)]
    # Optionally shuffle the indices to start filling from random positions
    if random_order:
        random.shuffle(indices)

    def solve_internal(index=0):
        print(f"Index {index}/{len(indices)}")
        if index == len(indices):
            if isValidSudoku3D_slices(B):
                return True  # All cells are filled correctly
            else:
                return False

        S = initialize_3d_solution_space(B)

        x, y, z = indices[index]
        if B[x, y, z] != 0:  # If already filled, skip to the next
            return solve_internal(index + 1)

        possible_numbers = [p for p, m in zip(possible_set, S[x, y, z, :]) if m]
        if random_try:
            random.shuffle(possible_numbers)  # Randomize numbers to increase the randomness of the solution
        for num in possible_numbers:
            if isValidSudoku3D_slices(B):
                B[x, y, z] = num
                update_plot(B, x, y, z, voxel_grid)
                if solve_internal(index + 1):
                    return True
                B[x, y, z] = 0  # Backtrack
                update_plot(B, x, y, z, voxel_grid)

        return False

    success = solve_internal()
    return B, success
