from sudoku_lib import * 
## =======================================================
# Generate an example 
g = 3

N = g**2
B = InitializeBoard(N)
S = InitializeSolutionSpace(B)

(B, Done, trialCount_BTG) = BacktrackMostConstrained(B, S,refreshCount= 1)

def add_one_after_g(vec, g):
    # Calculate how many times we need to add 1
    n_times = len(vec) // g
    for i in range(1, n_times+1):
        vec[i*g:] += 1
    return vec

def expand_sudoku(B):
    N = len(B)
    g = int(np.sqrt(N))
    r = np.array(range(g**2))
    r = add_one_after_g(r,g)

    B_new = InitializeBoard((g+1)**2)
    # Copy the old matrix values into the new one
    for i in range(B.shape[0]):
        for j in range(B.shape[1]):
            B_new[r[i],r[j]] = B[i,j]
            print_boardState(B_new,'.',True)
    return B_new

B = expand_sudoku(B)
S = InitializeSolutionSpace(B)
(B) = solve(B, S,dispBoard=True)
(B, Done, trialCount_BTG) = BacktrackMostConstrained(B, S,refreshCount= 1)
(B) = backTrackGenerate(B,refreshCount= 1000)