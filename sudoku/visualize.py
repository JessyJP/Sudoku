import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.colors import hsv_to_rgb

def print_board_state(board, substituteZero='', border=False, clear=True):
    """
    Prints the state of a 2D game board with optional borders and zero substitution.

    This function displays a square board where each row of the board is printed on a new line.
    Optionally, it can substitute zeros with a specified character, add borders around and within the grid,
    and clear the screen before printing to provide a fresh view of the board state.

    Parameters:
    - board (list[list[int]] or np.ndarray): The game board as a 2D list or a numpy array.
    - substituteZero (str, optional): The string used to replace zeros in the board's output. Defaults to an empty string, which means zeros will be shown as '0'.
    - border (bool, optional): Whether to print the board with borders to separate the cells and grid blocks. Defaults to False.
    - clear (bool, optional): Whether to clear the console screen before printing the board. Defaults to True.

    Returns:
    - None: This function does not return anything but prints the board to the standard output.
    """
    N = len(board)  # Number of rows (assuming square layers)
    g = int(np.sqrt(N))  # Calculate the size of each block within the board
    eS = len(str(abs(N)))  # Calculate the space needed to print the largest number

    if clear:
        os.system('cls' if os.name == 'nt' else 'clear')  # Clear the console screen for Windows or other systems

    if g**2 != N:
        print("Warning: The board size is not a perfect square. This may not be a valid Sudoku board configuration.")

    border_line = '-' * ((eS + 1) * N + 2 * g + 1)  # Construct the border line based on the board size and digit space

    for i, row in enumerate(board):
        if border and i % g == 0:  # Print the top border of each block
            print(border_line)

        print_row = []
        if border:
            print_row.append('|')  # Start border for each row
        for j, item in enumerate(row):
            # Print each item, substituting '0' with substituteZero if needed
            if item == 0 and substituteZero != '':
                print_row.append(substituteZero.rjust(eS))
            else:
                print_row.append(str(item).rjust(eS))
            if border and (j + 1) % g == 0 and (j + 1) != len(row):
                print_row.append('|')  # Add a vertical border between blocks within a row

        if border:
            print_row.append('|')  # End border for each row
        print(' '.join(print_row))

    if border:
        print(border_line)  # Print the bottom border of the last block


def print_solutionSpace(board, substituteZero='', border=False, clear=True):
    """
    Prints the state of a 2D game board's 3D solution space with multiple potential values per cell,
    with optional borders and zero substitution.

    This function displays a cube where each element of the cube may contain multiple potential values.
    Optionally, it can substitute zeros with a specified character, add borders around and within the grid blocks,
    and clear the screen before printing to provide a clear view of the solution space.

    Parameters:
    - board (list[list[list[int]]] or np.ndarray): The game board as a 3D list or a numpy array,
      where each innermost list contains possible values for that cell.
    - substituteZero (str, optional): The string used to replace zeros in the board's output.
      Defaults to an empty string, which means zeros will be shown as '0'.
    - border (bool, optional): Whether to print the board with borders to separate the cells and blocks.
      Defaults to False.
    - clear (bool, optional): Whether to clear the console screen before printing the board.
      Defaults to True.

    Returns:
    - None: This function does not return anything but prints the board to the standard output.
    """
    N = len(board)  # Number of outer blocks or layers
    g = int(np.sqrt(N))  # Calculate the size of each block within the board
    eS = len(str(max(item for subarray in board for subsubarray in subarray for item in subsubarray)))  # Calculate space needed

    if clear:
        os.system('cls' if os.name == 'nt' else 'clear')  # Clear the console screen for Windows or other systems

    if g ** 2 != N:
        print("Warning: The board size is not a perfect square. This may not be a valid Sudoku board configuration.")

    single_border_line = '-' * ((eS + 1) * N * g + 2 * (N + g) - 1)  # Single borderline
    double_border_line = '=' * ((eS + 1) * N * g + 2 * (N + g) - 1)  # Double borderline for more distinction

    for outer_x, outer_row in enumerate(board):
        if border and outer_x % g == 0:
            print(double_border_line)  # Print double line at the start of a new block

        for inner_x in range(g):
            print_row = []
            if border:
                print_row.append('|')  # Start border for each row within the block
            for outer_y, outer_col in enumerate(outer_row):
                for inner_y in range(g):
                    item = outer_col[inner_x * g + inner_y]
                    if item:
                        print_row.append(str(inner_x * g + inner_y + 1).rjust(eS))
                    else:
                        print_row.append(substituteZero.rjust(eS))  # Substitute zero if specified

                    if border and (inner_y + 1) % g == 0 and (inner_y + 1) < len(outer_col):
                        print_row.append('|')  # Vertical border between cells

                if border and (outer_y + 1) % g == 0 and (outer_y + 1) < len(outer_row):
                    print_row.append('|')  # Vertical border between blocks

            print(' '.join(print_row))
            if border and (inner_x + 1) % g == 0 and (inner_x + 1) < len(outer_row):
                print(single_border_line)  # Horizontal border between rows

    if border:
        print(double_border_line)  # Print the final border line at the bottom


class NumberVoxel:
    # Shared figure and axis setup correctly for 3D plotting
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    def __init__(self, index_coordinate, value=0, font_size=10, color='red'):
        """
        Initialize a NumberVoxel with a given position and optional value.

        Parameters:
        - index_coordinate (tuple): A tuple of three integers (x, y, z) indicating the voxel's position in 3D space.
        - value (int, optional): The initial numerical value to display in the voxel.
        - font_size (int, optional): The initial font size of the number.
        - color (str, optional): The initial color of the number text.
        """
        self.index_coordinate = index_coordinate
        self.value = value
        self.font_size = font_size
        self.color = color
        self.vertices = self.calculate_vertices()
        self.text = None
        self.lines = []  # Store handles for line objects
        self.print()  # Initial print when a voxel is created

    def calculate_vertices(self):
        """
        Calculate the vertices of the voxel based on its index coordinates.
        """
        x, y, z = self.index_coordinate
        return [(x, y, z), (x, y + 1, z), (x + 1, y + 1, z), (x + 1, y, z),
                (x, y, z + 1), (x, y + 1, z + 1), (x + 1, y + 1, z + 1), (x + 1, y, z + 1)]

    def print(self):
        """
        Print or reprint the voxel in 3D space, including its border and the number in the middle.
        """
        self.clear()  # Clear any previous voxel graphics
        # self.draw_cube()

        # Add the value in the middle of the voxel
        mid_x = self.index_coordinate[0] + 0.5
        mid_y = self.index_coordinate[1] + 0.5
        mid_z = self.index_coordinate[2] + 0.5
        self.text = self.ax.text(mid_x, mid_y, mid_z, str(self.value), color=self.color, fontsize=self.font_size, ha='center', va='center')
        plt.draw()

    def draw_cube(self):
        """
        Draw the cube based on the vertices.
        """
        pairs = [
            (0, 1), (1, 2), (2, 3), (3, 0),
            (4, 5), (5, 6), (6, 7), (7, 4),
            (0, 4), (1, 5), (2, 6), (3, 7)
        ]
        for p1, p2 in pairs:
            line = self.ax.plot([self.vertices[p1][0], self.vertices[p2][0]],
                                [self.vertices[p1][1], self.vertices[p2][1]],
                                [self.vertices[p1][2], self.vertices[p2][2]], 'b-')[0]
            self.lines.append(line)

    def update(self, value=None, font_size=None, color=None):
        """
        Update the properties of the voxel.

        Parameters:
        - value (int, optional): The new value to display in the voxel.
        - font_size (int, optional): The new font size of the number.
        - color (str, optional): The new color of the number text.
        """
        if value is not None:
            self.value = value
        if font_size is not None:
            self.font_size = font_size
        if color is not None:
            self.color = color
        if self.text:
            self.text.set_text(str(self.value))
            self.text.set_fontsize(self.font_size)
            self.text.set_color(self.color)

    def clear(self):
        """
        Clear the existing graphic elements for this number voxel.
        """
        if self.text:
            self.text.remove()
        for line in self.lines:
            line.remove()
        self.lines = []

# Ensure plot display is handled correctly
plt.ion()  # Turn on interactive mode, if needed


def plot_board_state(board, substituteZero='', border=True, clear=True):
    N = len(board)
    g = int(np.sqrt(N))
    voxel_grid = np.empty((N, N), dtype=object)  # Using a NumPy array for structured storage

    if clear:
        NumberVoxel.ax.cla()  # Clear the plot if needed

    # Draw a single bounding box around the entire grid
    draw_outer_border(board)

    for i in range(N):
        for j in range(N):
            value = board[i][j] if board[i][j] != 0 else substituteZero
            if voxel_grid[i, j] is None or clear:
                voxel_grid[i, j] = NumberVoxel((j, i, 0), value=str(value), color='red' if value != substituteZero else 'red', font_size=10)
            else:
                voxel_grid[i, j].update(value=str(value))  # Update existing voxels if not clearing
            plt.show()
    if border:
        for voxel in np.nditer(voxel_grid, flags=['refs_ok']):
            if voxel.item() is not None:
                voxel.item().draw_cube()

    plt.show()
    return voxel_grid


def draw_outer_border(board):
    """
    Draw a single bounding box around the board and adjust the viewing limits based on the board's dimensions.
    Labels are added to indicate rows, columns, and depth.

    Parameters:
    - board (numpy.ndarray): The game board, which can be 2D or 3D.
    """
    # Determine the dimensions of the board
    if board.ndim == 2:
        X, Y = board.shape
        Z = 1  # Default to 1 for 2D boards
    elif board.ndim == 3:
        X, Y, Z = board.shape
    else:
        raise ValueError("Unsupported number of dimensions for the board.")

    # Coordinates for the corners of the bounding box
    points = np.array([
        [0, 0, 0],
        [X, 0, 0],
        [X, Y, 0],
        [0, Y, 0],
        [0, 0, Z],
        [X, 0, Z],
        [X, Y, Z],
        [0, Y, Z]
    ])

    # Define the sides of the box, each as a list of vertices
    edges = [
        [points[0], points[1], points[5], points[4]],
        [points[0], points[3], points[7], points[4]],
        [points[2], points[1], points[5], points[6]],
        [points[2], points[3], points[7], points[6]],
        [points[0], points[1], points[2], points[3]],
        [points[4], points[5], points[6], points[7]],
    ]
    border_collection = Poly3DCollection(edges, linewidths=1.2, edgecolors='r', alpha=0.2)
    NumberVoxel.ax.add_collection3d(border_collection)
    ax = NumberVoxel.ax

    # Set the viewing limits for the plot based on the limits plus some margin
    margin = 1  # Additional space around the grid
    ax.set_xlim([0 - margin, X + margin])
    ax.set_ylim([0 - margin, Y + margin])
    ax.set_zlim([0 - margin, Z + margin])

    # Add labels for rows, columns, and depth
    ax.set_xlabel("Rows", color='red', fontsize=12)
    ax.set_ylabel("Columns", color='green', fontsize=12)
    ax.set_zlabel("Depth", color='blue', fontsize=12)

    # Optional: Adjust the aspect ratio for better viewing
    ax.set_box_aspect([X, Y, Z])  # Adjust aspect ratio to fit the dimensions

    # Update the plot
    plt.draw()

# Additional colouring methods
def create_color_map(N):
    """ Create a colormap with N distinct colors. """
    cmap = plt.get_cmap('viridis', N)  # Using 'viridis' colormap, you can choose any available colormap
    return cmap

def create_contrasting_colors(N):
    """ Generate N contrasting colors. """
    colors = []
    for i in range(N):
        hue = i / N  # Vary hue uniformly around the color wheel
        saturation = 0.9  # High saturation for vivid colors
        lightness = 0.5  # Mid lightness to avoid too dark or too light
        colors.append(hsv_to_rgb([hue, saturation, lightness]))
    return colors

def update_voxel_colors(voxel_grid, B):
    """ Update colors of a 3D voxel grid based on the dimension N. """
    N = len(B)
    cmap = create_contrasting_colors(N)
    size = voxel_grid.shape[0]  # Assuming voxel_grid is a N x N x N numpy array of Voxel objects

    for x in range(N):
        for y in range(N):
            for z in range(N):
                # color_index = (x * size * size + y * size + z) % N  # Calculate color index
                color = cmap[B[x,y,z]-1]  # Get color from colormap
                voxel_grid[x, y, z].update(color=color)