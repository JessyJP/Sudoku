import numpy as np
from sudoku.visualize import print_board_state, print_solutionSpace, NumberVoxel, plot_board_state
from sudoku3d.visualize_3d import print_full_solution_space
from utility_testing_init_methods import create_board_and_space_2d, create_board_and_space_3d
import matplotlib.pyplot as plt

def Test2DPrintFunctions():
    for g in [1, 2, 3]:
        board, solution_space = create_board_and_space_2d(g)
        print(f"\n{'-'*30}\nDisplay board with g={g}\n{'-'*30}\n")
        print_board_state(board, '.', True, False)
        print_solutionSpace(solution_space, '-', True, False)

def test_print_functions_3d():
    """
    Test the print functionality for 3D game boards and solution spaces across different group sizes.

    Group sizes tested are 1, 2, and 3, which correspond to 1x1x1, 4x4x4, and 9x9x9 grids respectively.
    Prints the board and solution space for each group size to demonstrate the functionality.
    """
    group_sizes = [1, 2, 3]  # Define the group sizes to test
    for g in group_sizes:
        print(f"Testing for group size {g} (Grid size {g**2}x{g**2}x{g**2}):")
        board, solution_space = create_board_and_space_3d(g)

        # Printing the full solution space for visualization
        print("Full Solution Space:")
        print_full_solution_space(solution_space, substituteZero='.', border=True, clear=False)
        print("\n" + "=" * 80 + "\n")

def print_single_voxel_test():
    voxel = NumberVoxel((0, 0, 0), 5)
    voxel.print()
    plt.pause(1)  # Pause for 1 second to view changes
    voxel.update(8)
    plt.pause(1)  # Pause again after updating
    voxel.update(value=8, font_size=12, color='blue')
    plt.pause(1)  # Final pause to view all changes

def plot_3d_board():
    board = np.random.randint(0, 10, (9, 9))  # Random board for demonstration
    plot_board_state(board, substituteZero='.', border=True, clear=True)
    plt.pause(1)  # Pause to view the board state

if __name__ == "__main__":
    plt.ion()  # Ensure the plot updates are displayed properly
    Test2DPrintFunctions()
    test_print_functions_3d()
    print_single_voxel_test()
    plot_3d_board()
    plt.show(block=True)
