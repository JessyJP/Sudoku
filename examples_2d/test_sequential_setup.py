from sudoku.solver import *  # Ensure you have detailed your imports elsewhere or here

# ==============================================================================
# Example to demonstrate the most constrained backtracking generation of Sudoku
# ==============================================================================


def BacktrackMostConstrained2(B, S, refreshCount=0, countLimit=0, trialCount_BTG=0):
    """
    Enhanced backtracking method that imposes additional constraints to efficiently solve Sudoku.

    Args:
        B (array): The Sudoku board state.
        S (array): The Sudoku solution space.
        refreshCount (int): How often to update the display of the board.
        countLimit (int): Upper limit on the number of trials.
        trialCount_BTG (int): Current count of backtracking trials.

    Returns:
        tuple: Returns a tuple containing the board, success status, and trial count.
    """
    if  countLimit > 0 and trialCount_BTG > countLimit:
        return (B, False, trialCount_BTG)

    def isValid(B,P):
        return not( P == 0 and B == 0 )

    if np.all(B > 0) and isValidSudoku(B):
        print_board_state(B, substituteZero='.', border=True)
        print("Success")
        print(f"Trial count: {trialCount_BTG}")
        return (B, True, trialCount_BTG)
    
    if not isValidSudoku(B):
        return (B, False, trialCount_BTG)
    
    N = len(B)
    val = int(np.sum(B > 0) / N)

    P = S[:,:,val];# <<<<<<<<<<<<<<<<-------------------------

    if np.all(P == False):
        return (B, False, trialCount_BTG)

    (rs, cs) = np.where(P == True)
    if rs.size > 0:
        r = rs[0]
        c = cs[0]
         
        # Stack R and C together and transpose
        RC = np.vstack((rs, cs)).T
        # Shuffle RC
        np.random.shuffle(RC)
        # Decompose RC back into R and C
        rs, cs = RC.T
        val += 1  #Adjust

        for i in range(len(rs)):
            r = rs[i];
            c = cs[i];
            trialCount_BTG = trialCount_BTG+1;
            B_cp = B.copy()
            S_cp = S.copy()
            
            if (B_cp[r,c] == 0):
                B_cp[r,c] = val;
                S_cp = squareOp(S_cp,B_cp,r,c)
                # keepChecking=True
            #end
            if ((refreshCount> 0) and not(trialCount_BTG % refreshCount)):
                print_board_state(B, '.', True)
                print(f" Trial count : {trialCount_BTG}")
            #end
            # solve(B,S,dispBoard=True)
            if refreshCount > 0:
                refreshCount = 1 if trialCount_BTG < 300 else int(np.log2(trialCount_BTG))**2
            #end
            (B_Done, res, trialCount_BTG) = BacktrackMostConstrained2(B_cp, S_cp, refreshCount=refreshCount, trialCount_BTG=trialCount_BTG)
            if res:
                return (B_Done , True , trialCount_BTG)
            #end
        #end
    #end

    return (B, False, trialCount_BTG)
#end


## ======================================================================
# Example Execution Section
## ======================================================================

# Setting up the initial conditions for Sudoku solving
g = 3
N = g**2
B = InitializeBoard(N)
S = initialize_solution_space(B)

# Initial display before solving
print_board_state(B, border=True)

# Attempt to solve the Sudoku and print the results
B, Done, trialCount_BTG = BacktrackMostConstrained2(B, S, refreshCount=1, countLimit=10**6)
print(f"Final trial count: {trialCount_BTG}")