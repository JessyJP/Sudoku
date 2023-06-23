from sudoku_lib import * 
## =======================================================
# Generate an example 
g = 5
N = g**2
B = InitializeBoard(N)
S = InitializeSolutionSpace(B)

trialCount = 0;
print_boardState(B,border=True)
(Done, B) = BacktrackMostConstrained(B, S, 1)

print_boardState(B,'.',True)
print(f" Trial count : {trialCount}")


if isValidSudoku(B) and count_non_zero(B) == N**2:
    print("NEW SUDOKU board created! : Success!")
    save_sudoku_board(B)
else:
    print("ERROR: PROBLEM occurred while creating a board!")
#end
