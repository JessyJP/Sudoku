from sudoku import *

## =======================================================
# Generate an example
g = 4
N = g**2
B = InitializeBoard(N)
S = initialize_solution_space(B)

# Generate index order
RC_zigzag = get_zigzag_matrix_indices(N)
RC_diag_block1 = get_diagonal_block_indices(g)    
RC_diag_block2 = get_diagonal_block_indices(g, reverse=True)    
R_linear   = get_linear_coordinates(N)
RC = merge_unique_ordered(RC_diag_block1, RC_diag_block2)
# RC = merge_unique_ordered(RC, R_linear)
elementCorrdinateVectors = coordinates_to_vectors(RC)


(B,trialCount) = backTrackGenerate(B,refreshCount=1,indexVectors=elementCorrdinateVectors);

S = initialize_solution_space(B)
B = solve(B,S,dispBoard=True)

RC = merge_unique_ordered(RC, R_linear)
elementCorrdinateVectors = coordinates_to_vectors(RC)
(B,trialCount) = backTrackGenerate(B,refreshCount=100,indexVectors=elementCorrdinateVectors);

print_board_state(B, '.', True)
print(f" Trial count : {trialCount}")


if isValidSudoku(B) and count_non_zero(B) == N**2:
    print("NEW SUDOKU board created! : Success!")
    save_sudoku_board(B)
else:
    print("ERROR: PROBLEM occurred while creating a board!")
#end
