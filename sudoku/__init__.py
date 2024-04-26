# Import specific functions from indexing_lib and other modules if needed
from .indexing import (
    coordinates_to_vectors,
    merge_unique_ordered,
    get_linear_element_indices_of_non_zero,
    get_linear_coordinates,
    get_random_element_indices_of_non_zero,
    get_zigzag_matrix_indices,
    get_block_indices,
    get_diagonal_block_indices,
)

from .solver import *

from .export import *
from .visualize import *

from .utilities import count_non_zero, get_linear_element_indices_of_non_zero, compare_boards

# You can also import functions from solver.py if needed
# from .solver import some_solver_function
