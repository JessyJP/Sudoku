import numpy as np
from sudoku.indexing import merge_unique_ordered, get_zigzag_matrix_indices

# ==========================================================================
# Conversion Functions


def coordinates_to_vectors_3d(coordinates):
    """
    Convert a list of (x, y, z) coordinates into separate lists of x, y, and z indices.

    Parameters:
    - coordinates: A list of tuples, where each tuple represents the (x, y, z) coordinates.

    Returns:
    - Three lists: one for x indices, one for y indices, and one for z indices.
    """
    x_indices = [coord[0] for coord in coordinates]
    y_indices = [coord[1] for coord in coordinates]
    z_indices = [coord[2] for coord in coordinates]
    return x_indices, y_indices, z_indices


# ==========================================================================
# Board Analysis Functions

def get_linear_element_indices_of_non_zero_3d(B):
    """
    Find the indices of non-zero elements in a 3D numpy array, typically representing a 3D grid.

    Parameters:
    - B: A 3D numpy array.

    Returns:
    - A tuple of three arrays: x, y, and z indices of non-zero elements.
    """
    (X, Y, Z) = np.where(B != 0)  # Rows, Columns, Depth
    return (X, Y, Z)


def get_linear_coordinates_3d(N, order='xyz'):
    """
    Generate linear coordinates for a cubical grid of size N, with various ordering options.

    Parameters:
    - N: The size of one side of the cubical grid.
    - order: A string representing the order of coordinates ('xyz', 'xzy', 'yxz', etc.).

    Returns:
    - A list of tuples representing the coordinates.
    """
    if order == 'xyz':
        return [(x, y, z) for x in range(N) for y in range(N) for z in range(N)]
    elif order == 'xzy':
        return [(x, z, y) for x in range(N) for z in range(N) for y in range(N)]
    elif order == 'yxz':
        return [(y, x, z) for y in range(N) for x in range(N) for z in range(N)]
    # Add other orders as needed


def get_random_element_indices_of_non_zero_3d(B):
    """
    Randomly shuffle the indices of non-zero elements in a 3D numpy array.

    Parameters:
    - B: A 3D numpy array.

    Returns:
    - Three arrays: x, y, and z indices of non-zero elements, shuffled.
    """
    (X, Y, Z) = np.where(B != 0)
    XYZ = np.vstack((X, Y, Z)).T
    np.random.shuffle(XYZ)
    X, Y, Z = XYZ.T
    return (X, Y, Z)


def get_zigzag_matrix_indices_3d(n):
    """
    Generate indices for traversing a cubic matrix of size n in a zigzag manner across each layer.

    Parameters:
    - n: The size of the cubic matrix.

    Returns:
    - A list of tuples representing the indices to traverse the matrix in a zigzag pattern.
    """
    indices = []
    for z in range(n):
        layer_indices = get_zigzag_matrix_indices(n)  # Use existing 2D method within each layer
        indices.extend([(x, y, z) for (x, y) in layer_indices])
    return indices


def get_block_indices_3d(block_x, block_y, block_z, g):
    """
    Get the indices of a specific block in a cubic grid.

    Parameters:
    - block_x: The x index of the block (0-indexed).
    - block_y: The y index of the block (0-indexed).
    - block_z: The z index of the block (0-indexed).
    - g: The size of the block.

    Returns:
    - A list of tuples representing the indices within the specified block.
    """
    start_x, start_y, start_z = g * block_x, g * block_y, g * block_z
    return [(x, y, z) for x in range(start_x, start_x + g) for y in range(start_y, start_y + g) for z in range(start_z, start_z + g)]


def get_diagonal_block_indices_3d(g, reverse=False):
    """
    Get the indices of blocks along the diagonal of a cubic grid, either the main diagonal or the reverse diagonal.

    Parameters:
    - g: The size of the blocks.
    - reverse: If True, get indices for the reverse diagonal; otherwise, the main diagonal.

    Returns:
    - A list of tuples representing the indices within the diagonal blocks.
    """
    diagonal_blocks_indices = []
    for i in range(g):
        if reverse:
            diagonal_blocks_indices.extend(get_block_indices_3d(i, g - i - 1, i, g))
        else:
            diagonal_blocks_indices.extend(get_block_indices_3d(i, i, i, g))
    return diagonal_blocks_indices
