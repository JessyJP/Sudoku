import os
import numpy as np
from matplotlib import pyplot as plt

from sudoku import print_board_state, print_solutionSpace, NumberVoxel
from sudoku.visualize import draw_outer_border


def print_layer(board, layer_index, substituteZero='', border=False, clear=True):
    """
    Prints a single layer of a 3D game board with optional borders and zero substitution.

    Parameters:
    - board: A 3D numpy array representing the game board.
    - layer_index: The index of the layer to print.
    - substituteZero: The string used to replace zeros in the board's output. Defaults to an empty string.
    - border: Whether to print the board with borders to separate the cells and grid blocks. Defaults to False.
    - clear: Whether to clear the console screen before printing the board. Defaults to True.
    """
    print_board_state(board[layer_index], substituteZero=substituteZero, border=border, clear=clear)


def print_layers(board, substituteZero='', border=False, clear=True):
    """
    Prints all layers of a 3D game board, one by one, with optional borders and zero substitution.

    Parameters:
    - board: A 3D numpy array representing the game board.
    - substituteZero: The string used to replace zeros in the board's output. Defaults to an empty string.
    - border: Whether to print the board with borders to separate the cells and grid blocks. Defaults to False.
    - clear: Whether to clear the console screen before printing the board. Defaults to True.
    """
    depth = board.shape[2]  # Assuming board is at least 3D
    if clear:
        os.system('cls' if os.name == 'nt' else 'clear')
    for i in range(depth):
        print(f"Layer {i+1} out of {depth}:")
        print_layer(board, i, substituteZero=substituteZero, border=border, clear=False)
        if i < board.shape[2] - 1:
            print("\n" + ("=" * 40) + "\n")


def print_solution_space_layer(solution_space, layer_index, substituteZero='', border=False, clear=True):
    """
    Prints a single layer of a 3D game board's solution space, allowing visualization of potential values in each cell.

    Parameters:
    - solution_space: A 4D numpy array where each cell at a given depth contains possible values for that cell.
    - layer_index: Index of the layer to print within the solution space.
    - substituteZero: String used to replace zeros in the board's output, enhancing clarity.
    - border: Whether to add borders around cells and blocks for better visual segmentation.
    - clear: Whether to clear the console before printing, ensuring visibility of only the current state.

    Returns:
    - None: This function does not return anything but prints the specified layer to the standard output.
    """
    # TODO: use  -> print_solutionSpace
    print_solutionSpace(solution_space[:, :, layer_index, :], substituteZero=substituteZero, border=border, clear=clear)


def print_full_solution_space(solution_space, substituteZero='', border=False, clear=True):
    """
    Prints the entire solution space of a 3D game board, layer by layer, showcasing potential values for each cell.

    Parameters:
    - solution_space: A 4D numpy array where each cell at any depth lists potential values.
    - substituteZero: String to replace zeros in the output, enhancing readability where no potential values exist.
    - border: Whether to add borders to enhance visual segmentation of cells and blocks.
    - clear: Whether to clear the console screen before printing, to maintain a clean visual output.

    Returns:
    - None: This function does not return anything but provides a comprehensive view of the solution space.
    """
    if clear:
        os.system('cls' if os.name == 'nt' else 'clear')  # Clear the console screen

    for layer_index in range(solution_space.shape[2]):
        print(f"Layer {layer_index + 1}/{solution_space.shape[2]}:")
        print_solution_space_layer(solution_space, layer_index, substituteZero, border, False)
        print("\n" + ("." * 80) + "\n")


def plot_board_state_3d(board, substituteZero='', border=True, clear=True):
    N = len(board)
    g = int(np.sqrt(N))
    voxel_grid = np.empty((N, N, N), dtype=object)  # Using a NumPy array for structured storage

    if clear:
        NumberVoxel.ax.cla()  # Clear the plot if needed

    # Draw a single bounding box around the entire grid
    draw_outer_border(voxel_grid)

    for x in range(N):
        for y in range(N):
            for z in range(N):
                value = board[x][y][z] if board[x][y][z] != 0 else substituteZero
                if voxel_grid[x, y, z] is None or clear:
                    voxel_grid[x, y, z] = NumberVoxel((x, y, z), value=str(value), color='red' if value != substituteZero else 'red', font_size=10)
                else:
                    voxel_grid[x, y, z].update(value=str(value))  # Update existing voxels if not clearing
                plt.show()
    if border:
        for voxel in np.nditer(voxel_grid, flags=['refs_ok']):
            if voxel.item() is not None:
                voxel.item().draw_cube()

    plt.show()
    return voxel_grid

def update_plot_board_state_3d(board, voxel_grid, substituteZero='', border=True, clear=True):
    N = len(board)
    g = int(np.sqrt(N))
    if clear:
        NumberVoxel.ax.cla()  # Clear the plot if needed

    for x in range(N):
        for y in range(N):
            for z in range(N):
                value = board[x, y, z] if board[x, y, z] != 0 else substituteZero
                voxel_grid[x, y, z].update(value=value)

                if border and value != substituteZero:
                    voxel_grid[x, y, z].draw_cube()