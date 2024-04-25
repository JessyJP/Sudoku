import numpy as np


def create_board_and_space_3d(g):
    """
    Generates a 3D game board and a solution space for testing based on the specified group size.

    Parameters:
    - g (int): The size of the group, which determines the dimensions of the board and the solution space.

    Returns:
    - tuple: A tuple containing the generated board and solution space.
    """
    N = g ** 2
    board = np.arange(0, N ** 3).reshape(N, N, N) % N + 1  # Modulo operation to keep numbers within 1 to N
    num_possible_values = N  # Assuming the number of possible values could be up to N
    solution_space = np.zeros((N, N, N, num_possible_values), dtype=int)

    for i in range(N):
        for j in range(N):
            for k in range(N):
                possible_values = np.random.choice(range(1, num_possible_values + 1), num_possible_values, replace=False)
                solution_space[i, j, k, :] = possible_values

    return board, solution_space


def create_board_and_space_2d(g):
    """
    Generates a 2D game board and a solution space for testing based on the specified group size.

    Parameters:
    - g (int): The size of the group, which determines the dimensions of the board and the solution space.

    Returns:
    - tuple: A tuple containing the generated board and solution space, suitable for 2D puzzles like Sudoku.
    """
    N = g ** 2
    board = np.arange(0, N ** 2).reshape(N, N) % N + 1  # Modulo operation to keep numbers within 1 to N
    num_possible_values = N  # Assuming the number of possible values could be up to N
    solution_space = np.zeros((N, N, num_possible_values), dtype=int)

    for i in range(N):
        for j in range(N):
            possible_values = np.random.choice(range(1, num_possible_values + 1), num_possible_values, replace=False)
            solution_space[i, j, :] = possible_values

    return board, solution_space
