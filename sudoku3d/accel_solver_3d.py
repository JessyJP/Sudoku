from numba import jit, prange
import numpy as np
import random
from matplotlib import pyplot as plt


# ==========================================================
# Sudoku Library: Utilities for handling Sudoku puzzles
# ==========================================================
@jit(nopython=True)
def Initialize3DBoard(N):
    """
    Initialize a 3D board for Sudoku with all cells set to zero.
    """
    return np.zeros((N, N, N), dtype=np.int32)

@jit(nopython=True)
def initialize_3d_solution_space(B):
    """
    Initialize the solution space for 3D Sudoku with possible values for each cell.
    """
    N = len(B)
    S = np.full((N, N, N, N), True)
    for r in range(N):
        for c in range(N):
            for z in range(N):
                if B[r, c, z] > 0:
                    S = cubeOp(S, B, r, c, z)
    return S

@jit(nopython=True)
def squareOp(S, B, R, C):
    """
    Apply Sudoku rules to eliminate impossible values from the solution space.
    """
    if B[R, C] == 0:
        return S
    N = len(B)
    g = int(np.sqrt(N))
    val = B[R, C] - 1
    S[R, C, :] = False
    S[R, :, val] = False
    S[:, C, val] = False
    block_start_row = g * (R // g)
    block_start_col = g * (C // g)
    for r_ in range(block_start_row, block_start_row + g):
        for c_ in range(block_start_col, block_start_col + g):
            S[r_, c_, val] = False
    return S

@jit(nopython=True)
def cubeOp(S, B, r, c, z):
    """
    Apply Sudoku rules in a 3D cube to eliminate impossible values from the solution space.
    """
    N = B.shape[0]
    val = B[r, c, z] - 1
    block_size = int(np.sqrt(N))
    R0 = (r // block_size) * block_size
    C0 = (c // block_size) * block_size
    Z0 = (z // block_size) * block_size
    S[r, c, z, :] = False
    S[r, c, :, val] = False
    S[r, :, z, val] = False
    S[:, c, z, val] = False
    S[R0:R0 + block_size, C0:C0 + block_size, z, val] = False
    S[R0:R0 + block_size, c, Z0:Z0 + block_size, val] = False
    S[r, C0:C0 + block_size, Z0:Z0 + block_size, val] = False
    return S

@jit(nopython=True)
def isValidSudoku(B):
    """
    Check if the board is a valid Sudoku.
    """
    N = len(B)
    g = int(np.sqrt(N))
    row = np.zeros((N, N), dtype=np.int32)
    col = np.zeros((N, N), dtype=np.int32)
    block = np.zeros((N, N), dtype=np.int32)
    for r in range(N):
        for c in range(N):
            digit = B[r, c]
            if digit == 0:
                continue
            d = int(digit) - 1
            block_index = (r // g) * g + (c // g)
            if row[r, d] or col[c, d] or block[block_index, d]:
                return False
            row[r, d], col[c, d], block[block_index, d] = True, True, True
    return True

@jit(nopython=True)
def isValidSudoku3D_slices(board):
    """
    Check if a 3D Sudoku board is valid across all slices in three dimensions.
    """
    N = len(board)
    for z in prange(N):
        if not isValidSudoku(board[:, :, z]):
            return False
    for x in prange(N):
        if not isValidSudoku(board[x, :, :]):
            return False
    for y in prange(N):
        if not isValidSudoku(board[:, y, :]):
            return False
    return True

# ==========================================================
# Sudoku Library: Solver
# ==========================================================
@jit(nopython=True)
def setValueOnBoard(B,S,r,c,val):
    global keepChecking
    # Set value on the board
    val = val + 1  # Adjust
    if (B[r,c] == 0):
        B[r,c] = val;
        S = squareOp(S,B,r,c)
        keepChecking=True
    #end
    return (B,S)
#end

@jit(nopython=True)
def solve(B, S):
    """
    Attempt to solve the Sudoku puzzle by iterating and updating the board and solution space.
    """
    N = len(B)
    g = int(np.sqrt(N))
    keepChecking = True
    while keepChecking:
        keepChecking = False
        for i in range(N):
            bR = int(i/g)
            bC = i%g
            for val in range(N):
                if np.sum(S[g*bR:g*(bR+1), g*bC:g*(bC+1), :, val]) == 1:
                    r, c = np.where(S[g*bR:g*(bR+1), g*bC:g*(bC+1), :, val])
                    B[r[0]+g*bR, c[0]+g*bC] = val + 1
                    S = squareOp(S, B, r[0]+g*bR, c[0]+g*bC)
                    keepChecking = True
    return B

@jit(nopython=True)
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


def backTrackGenerate3D3(B, voxel_grid=None, refreshCount=1000, random_order=False, random_try=False, try_to_solve=False):
    N = len(B)  # Assuming B is N x N x N
    possible_set = list(range(1, N + 1))
    S = initialize_3d_solution_space(B)
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

        if try_to_solve and isValidSudoku3D_slices(B):
            B_tmp = B.copy()
            while True:
                try:
                    B_tmp, success = solve3D_by_all_slices(B_tmp, S)
                except:
                    return False
                if success:
                    if np.array_equal(B_tmp, B):
                        break
                    else:
                        B[B_tmp>0] = B_tmp[B_tmp>0]
                else:
                    break

        x, y, z = indices[index]
        if B[x, y, z] != 0:  # If already filled, skip to the next
            return solve_internal(index + 1)

        possible_numbers = [p for p, m in zip(possible_set, S[x, y, z, :]) if m]
        if random_try:
            random.shuffle(possible_numbers)  # Randomize numbers to increase the randomness of the solution
        for num in possible_numbers:
            if isValidSudoku3D_slices(B):
                B[x, y, z] = num
                if solve_internal(index + 1):
                    return True
                B[x, y, z] = 0  # Backtrack

        return False

    success = solve_internal()
    return B, success


