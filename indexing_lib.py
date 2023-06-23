import numpy as np



## =========================================================================
## Top conversion function
def coordinates_to_vectors(coordinates):
    x_indices = [coord[0] for coord in coordinates]
    y_indices = [coord[1] for coord in coordinates]
    return x_indices, y_indices

def merge_unique_ordered(list1, list2):
    seen = set()
    result = []
    
    for item in list1 + list2:
        if item not in seen:
            result.append(item)
            seen.add(item)
    return result

## =========================================================================
def get_Linear_Element_indices_ofNonZero(B):
    (R, C) = np.where(B == 0)
    return (R, C)


def get_linear_coordinates(N, column_first=False):
    if column_first:
        return [(j, i) for i in range(N) for j in range(N)]
    else:
        return [(i, j) for i in range(N) for j in range(N)]

def get_random_Element_indices_ofNonZero(B):
    (R, C) = np.where(B == 0)
    # Stack R and C together and transpose
    RC = np.vstack((R, C)).T
    # Shuffle RC
    np.random.shuffle(RC)
    # Decompose RC back into R and C
    R, C = RC.T
    return (R,C)


def get_zigzag_matrix_indices(n):
    # Initialize the indices
    indices = [(0, 0)]

    for diagonal in range(1, 2*n):
        if diagonal < n:
            # Traverse diagonally up
            if diagonal % 2 == 0:
                indices.extend([(i, diagonal-i) for i in range(diagonal+1)])
            # Traverse diagonally down
            else:
                indices.extend([(diagonal-i, i) for i in range(diagonal+1)])
        else:
            # Traverse diagonally down
            if diagonal % 2 == 0:
                indices.extend([(i, diagonal-i) for i in range(n-1, diagonal-n, -1)])
            # Traverse diagonally up
            else:
                indices.extend([(diagonal-i, i) for i in range(n-1, diagonal-n, -1)])
                
    return indices


def get_block_indices(block_row, block_col, g):
    # Note that block_row and block_col are 0-indexed
    # That is, the top left block is (0, 0), not (1, 1)
    
    start_row, start_col = g * block_row, g * block_col
    return [(i, j) for i in range(start_row, start_row + g) for j in range(start_col, start_col + g)]


def get_diagonal_block_indices(g,reverse=False):
    diagonal_blocks_indices = []
    for i in range(g):
        if reverse:
            diagonal_blocks_indices.extend(get_block_indices(g-i-1, i, g))
        else:
            diagonal_blocks_indices.extend(get_block_indices(i, i, g))

    return diagonal_blocks_indices



