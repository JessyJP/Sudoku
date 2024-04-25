from sudoku import *
## =======================================================
# Generate an example 
g = 5

N = g**2
B = InitializeBoard(N)
S = initialize_solution_space(B)


print_board_state(B, border=True)
(B, Done, trialCount_BTG) = BacktrackMostConstrained(B, S,refreshCount= 1,countLimit= 10**6)

print_board_state(B, '.', True)
print(f" Trial count : {trialCount_BTG}")


if isValidSudoku(B) and count_non_zero(B) == N**2:
    print("NEW SUDOKU board created! : Success!")
    save_sudoku_board(B)
else:
    print("ERROR: PROBLEM occurred while creating a board!")
#end
