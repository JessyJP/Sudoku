from sudoku3d.solver_3d import Initialize3DBoard

g = 3  # Define the cube root of N; for a 3D board, g could be 2, 3, 4, etc.
board_3d = Initialize3DBoard(g)
print(f"Initialized a 3D Sudoku board of dimensions {board_3d.shape}:\n{board_3d}")