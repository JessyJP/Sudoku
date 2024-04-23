import numpy as np


# ==========================================================================
# Conversion Functions
# ==========================================================================

def coordinates_to_vectors(coordinates):
    """
    Convert a list of (x, y) coordinates into separate lists of x and y indices.

    Parameters:
    - coordinates: A list of tuples, where each tuple represents the (x, y) coordinates.

    Returns:
    - Two lists: one for x indices and one for y indices.
    """
    x_indices = [coord[0] for coord in coordinates]  # Extract x coordinates
    y_indices = [coord[1] for coord in coordinates]  # Extract y coordinates
    return x_indices, y_indices


def merge_unique_ordered(list1, list2):
    """
    Merge two lists into one, maintaining the order and excluding duplicates.
    It first appends items from the first list and then from the second, skipping duplicates.

    Parameters:
    - list1: The first list.
    - list2: The second list.

    Returns:
    - A merged list without duplicates, preserving the order of appearance.
    """
    seen = set()  # Track seen items to avoid duplicates
    result = []

    for item in list1 + list2:  # Iterate over both lists
        if item not in seen:  # If an item is not in seen, append it to result
            result.append(item)
            seen.add(item)  # Mark item as seen
    return result


# ==========================================================================
# Board Analysis Functions
# ==========================================================================

def get_linear_element_indices_of_non_zero(B):
    """
    Find the indices of non-zero elements in a 2D numpy array, typically representing a Sudoku board.
    This function is often used to identify filled cells on the board.

    Parameters:
    - B: A 2D numpy array representing the Sudoku board.

    Returns:
    - A tuple of two arrays: rows and columns indices of non-zero elements.
    """
    (R, C) = np.where(B == 0)  # Find indices where elements are non-zero
    return (R, C)


def get_linear_coordinates(N, column_first=False):
    """
    Generate linear coordinates for a square grid of size N, either row-first or column-first.
    This can be used to iterate over the cells of a Sudoku board in different orders.

    Parameters:
    - N: The size of one side of the square grid.
    - column_first: If True, generate column coordinates first; otherwise, row coordinates first.

    Returns:
    - A list of tuples representing the coordinates.
    """
    if column_first:
        return [(j, i) for i in range(N) for j in range(N)]  # Column-first order
    else:
        return [(i, j) for i in range(N) for j in range(N)]  # Row-first order


def get_random_element_indices_of_non_zero(B):
    """
    Randomly shuffle the indices of non-zero elements in a 2D numpy array, typically a Sudoku board.
    This can be useful for randomizing the order of operations on the board's filled cells.

    Parameters:
    - B: A 2D numpy array representing the Sudoku board.

    Returns:
    - Two arrays: rows and columns indices of non-zero elements, shuffled.
    """
    (R, C) = np.where(B == 0)  # Find indices of non-zero elements
    RC = np.vstack((R, C)).T  # Stack R and C together and transpose
    np.random.shuffle(RC)  # Shuffle RC to randomize indices
    R, C = RC.T  # Decompose RC back into R and C
    return (R, C)


def get_zigzag_matrix_indices(n):
    """
    Generate indices for traversing a square matrix of size n in a zigzag manner.
    This pattern is useful for algorithms that require processing the matrix in a diagonal order.

    Parameters:
    - n: The size of the square matrix.

    Returns:
    - A list of tuples representing the indices to traverse the matrix in a zigzag pattern.
    """
    indices = [(0, 0)]  # Initialize with the top-left corner
    for diagonal in range(1, 2 * n):
        range_start = 0 if diagonal < n else diagonal - n + 1  # Start of diagonal range
        range_end = diagonal + 1 if diagonal < n else n  # End of diagonal range
        row_indices = range(range_start, range_end)  # Generate row indices for current diagonal

        # Zigzag traversal: switch between upward and downward diagonals
        if diagonal % 2 == 0:
            indices.extend([(i, diagonal - i) for i in row_indices])
        else:
            indices.extend([(diagonal - i, i) for i in row_indices])
    return indices


def get_block_indices(block_row, block_col, g):
    """
    Get the indices of a specific block in a Sudoku grid, where each block is a smaller square within the grid.
    Blocks are identified by their row and column indices, starting from the top-left corner.

    Parameters:
    - block_row: The row index of the block (0-indexed).
    - block_col: The column index of the block (0-indexed).
    - g: The size of the block (typically 3 for a standard 9x9 Sudoku).

    Returns:
    - A list of tuples representing the indices within the specified block.
    """
    # Note that block_row and block_col are 0-indexed
    # That is, the top left block is (0, 0), not (1, 1)
    start_row, start_col = g * block_row, g * block_col  # Calculate starting row and column
    return [(i, j) for i in range(start_row, start_row + g) for j in range(start_col, start_col + g)]


def get_diagonal_block_indices(g, reverse=False):
    """
    Get the indices of blocks along the diagonal of a Sudoku grid, either the main diagonal or the reverse diagonal.

    Parameters:
    - g: The size of the blocks (typically 3 for a standard 9x9 Sudoku).
    - reverse: If True, get indices for the reverse diagonal; otherwise, the main diagonal.

    Returns:
    - A list of tuples representing the indices within the diagonal blocks.
    """
    diagonal_blocks_indices = []
    for i in range(g):
        # Add indices for blocks along the specified diagonal
        if reverse:
            diagonal_blocks_indices.extend(get_block_indices(g - i - 1, i, g))
        else:
            diagonal_blocks_indices.extend(get_block_indices(i, i, g))
    return diagonal_blocks_indices
