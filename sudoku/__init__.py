# Import specific functions and classes from indexing, solver, export, and visualize modules

# From indexing.py
from .indexing import (
    coordinates_to_vectors,
    merge_unique_ordered,
    compare_boards,
    convert_to_int_board,
    count_non_zero,
    get_linear_element_indices_of_non_zero,
    get_linear_coordinates,
    get_random_element_indices_of_non_zero,
    get_zigzag_matrix_indices,
    get_block_indices,
    get_diagonal_block_indices
)

# From solver.py
from .solver import (
    InitializeBoard,
    initialize_solution_space,
    squareOp,
    isValidSudoku,
    solve,
    backTrackSolve,
    backTrackGenerate,
    BacktrackMostConstrained
)

# From export.py
from .export import (
    load_sudoku_board,
    save_sudoku_board,
    export_board_to_html,
    open_html,
    export_board_to_TEX,
    compilePDF,
    compile_TEX_to_PDF,
    open_pdf,
    cleanup_auxiliary_files
)

# From visualize.py
from .visualize import (
    print_board_state,
    print_solutionSpace,
    NumberVoxel,
    plot_board_state
)

from .make_game import (
    make_game_from_solution_sequentialy,
    make_game_from_solution
)

# Optional: Import additional classes or functions if more functionalities are developed
# from .additional_module import additional_function
