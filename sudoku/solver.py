import time
import numpy as np
import random
# Relative import of indexing_lib from the same package (sudoku)
from .indexing import get_linear_element_indices_of_non_zero
from .visualize import print_solutionSpace, print_board_state

"""
Sudoku Library (sudoku) - A collection of utilities for handling Sudoku puzzles.
"""

## =========================================================================

def InitializeBoard( N ):
    return  np.full((N, N), 0)
#end

## Solution Initialization function
def initialize_solution_space(B, dispSolutionState=False):#
    N = len(B);
    # Initialize the Full possible matrix
    iniVal = True; 
    # S = np.zeros((L, L, L))
    # S = [[[iniVal for k in range(L)] for j in range(L)] for i in range(L)]
    S = np.full((N, N, N), iniVal)

    # Remove all impossible options
    for r in range(N):
        for c in range(N):
            if (B[r,c] > 0):
                S = squareOp(S,B,r,c, dispSolutionState)
            #end
        #end
    #end
    return S
#end

# Square operation function
def squareOp(S,B,R,C,dispSolutionState = False):
    if B[R,C] == 0: return S;  #end

    N = len(B);# Get the board Length
    g = int(np.sqrt(N));

    X = B[R,C]-1;

    # Check Square
    for x in range(N):
        S[R,C,x] = False;
        if dispSolutionState: print_solutionSpace(S,' ',True)  #end
    #end

    # Check Rows
    for r_ in range(N):
        S[r_,C,X] = False;
        if dispSolutionState: print_solutionSpace(S,' ',True)  #end
    #end

    # Check Columns
    for c_ in range(N):
        S[R,c_,X] = False;
        if dispSolutionState: print_solutionSpace(S,' ',True) #end
    #end
        
    # Check Block
    for r_ in range(g*int(R/g),  g* (int(R/g) + 1) ):
        for c_ in range(g*int(C/g), g*(int(C/g) + 1) ):
            S[r_,c_,X] = False;
            if dispSolutionState: print_solutionSpace(S,' ',True)  #end
        #end
    #end

    if dispSolutionState: print_solutionSpace(S,' ',True)  #end
    return S
#end

# Check board validity
def isValidSudoku(B):
    N = len(B)
    g = int(np.sqrt(N))
    row = np.zeros((N, N), dtype=int)
    col = np.zeros((N, N), dtype=int)
    block = np.zeros((N, N), dtype=int)
    for r in range(N):
        for c in range(N):
            digit = B[r,c]
            if digit == 0:
                continue
            #end
            d = int(digit) - 1
            if d < 0:
                continue
            #end
            if row[r,d]:
                return False
            #end
            row[r,d] = True
            if col[c,d]:
                return False
            #end
            col[c,d] = True
            bInd =  (r//g)*g + c//g
            if block[bInd, d]:
                return False
            #end
            block[bInd,d] = True
        #end
    #end
    return True
#end

## =========================================================================
## Normal Solve with elimination

# Solve function
def solve(B,S,dispBoard=False,dispSolutionState=False):
    N = len(B);
    g = int(np.sqrt(N));
    keepChecking = True

    def setValueOnBoard(B,S,r,c,val):
        nonlocal keepChecking
        # Set value on the board
        val = val+1# Adjust
        if (B[r,c] == 0):
            B[r,c] = val;
            S = squareOp(S,B,r,c)
            keepChecking=True
        #end
        if dispBoard: print_board_state(B, '.', True) #end
        if dispSolutionState: print_solutionSpace(S,' ',True) #end 
        return (B,S)
    #end

    while(keepChecking):
        keepChecking = False
        try:
            del rs ,cs , vals
        except:
            pass

        ## Check Blocks with solutions that have only 1 option for a given number |limited: 1st DIM & 2nd DIM
        for i in range(N):
            # Get block solution sub-matrix
            bR = int(i/g); bC = i%g;     
            vals = np.sum(np.sum(S[g*bR:g*bR+3,g*bC:g*bC+3,:] , axis=0), axis=0) == 1;
            if sum(vals):
                vals = np.where( vals.squeeze()  ) # Get indices
                for val in vals:
                    val = val[0];
                    r,c = np.where(S[g*bR:g*bR+3,g*bC:g*bC+3,val].squeeze())
                    # Get triplet 0f indices and value           
                    r = r[0]+g*bR; c = c[0]+g*bC;
                    # val = list(S[r,c,:]).index(True);
                    (B,S) = setValueOnBoard(B,S,r,c,val)
                #end
            #end
        #end

        ## Check all solutions that have only 1 option left |3rd DIM
        (rs,cs) = np.where( np.sum(S, axis=2) == 1  )  
        for i in range(rs.size): 
            # Get triplet 0f indices and value           
            r = rs[i];
            c = cs[i];
            val = list(S[r,c,:]).index(True);
            (B,S) = setValueOnBoard(B,S,r,c,val)
        #end

        ## Check columns with solutions that have only 1 option for a given number |2nd DIM
        (rs,vals) = np.where( np.sum(S, axis=1) == 1  )   
        for i in range(rs.size):   
            # Get triplet 0f indices and value         
            r = rs[i];
            val = vals[i]
            c = list( S[r,:,val] ).index(True);
            (B,S) = setValueOnBoard(B,S,r,c,val)
        #end

        ## Check rows with solutions that have only 1 option for a given number |1st DIM
        (cs,vals) = np.where( np.sum(S, axis=0) == 1  )   
        for i in range(cs.size):   
            # Get triplet 0f indices and value         
            c = cs[i];
            val = vals[i]
            r = list( S[:,c,val] ).index(True);
            (B,S) = setValueOnBoard(B,S,r,c,val)
        #end

    return B
#end

## =========================================================================
## Backtrack solve

def backTrackSolve(B, indexVectors=None, refreshCount=0):
    trialCount = 0;
    N = len(B)
    (R, C) = get_linear_element_indices_of_non_zero(B)

    if indexVectors is not None:
        (R, C) = indexVectors;

    # S = InitializeSolutionSpace(B)

    def backTrackSolve(depth):
        nonlocal B, trialCount, refreshCount , R , C
        if depth >=  len(R):
            return True;
        #end

        r = R[depth]; c = C[depth];
        while( B[r,c] > 0 ):
            depth = depth +1;
            if depth >= N**2:
                if isValidSudoku(B): return True;
                else: return False
            #end
            r = R[depth]; c = C[depth];
        
        if isValidSudoku(B) and not len(R):
            return True
        #end
        solutionFound = False;
        S = initialize_solution_space(B);
        possibleValues = range(1, N+1)
        possibleValues = np.where(S[r,c,:])[0]+1
        randomPermutation = list(possibleValues)
        random.shuffle(randomPermutation) 
        for d in randomPermutation:
            trialCount = trialCount+1;

            B[r,c] = str(d)
            
            if not(trialCount % refreshCount):
                print_board_state(B, '.', True)
                print(f" Trial count : {trialCount}")
            #end

            if isValidSudoku(B):
                solutionFound = backTrackSolve(depth+1)
                if solutionFound or not len(R):
                    return True
                else:
                    B[r,c] = 0;
                #end
            else:
                B[r,c] = 0;
            #end
        #end
    #end

    backTrackSolve(0)
    return (B,trialCount)
#end

## =========================================================================
## Backtrack generate

def backTrackGenerate(B, indexVectors=None, refreshCount=0):
    trialCount = 0;
    N = len(B)
    (R, C) = get_linear_element_indices_of_non_zero(B)

    if indexVectors is not None:
        (R, C) = indexVectors;
        
    # S = InitializeSolutionSpace(B)

    def backTrackSolve(depth):
        nonlocal B, trialCount, refreshCount , R , C
        if depth >= N**2 or depth >= len(R):
            return True;
        #end

        def clearBdepth(depth):
            nonlocal B, N
            # Define the linear indices
            for i in range(depth, N**2):
                r = R[i]; c = C[i]; 
                B[r,c] = B[r,c]*0;
            #end
            print_board_state(B, '.', True)
        #end

        r = R[depth]; c = C[depth];
        while( B[r,c] > 0 ):
            depth = depth +1;
            if depth >= N**2:
                if isValidSudoku(B): return True;
                else: return False
            #end
            r = R[depth]; c = C[depth];
        
        if isValidSudoku(B) and not len(R):
            return True
        #end
        
        solutionFound = False;
        S = initialize_solution_space(B);
        possibleValues = range(1, N+1)
        possibleValues = np.where(S[r,c,:])[0]+1
        randomPermutation = list(possibleValues)
        # random.shuffle(randomPermutation) 
        for d in randomPermutation:
            trialCount = trialCount+1;

            B[r,c] = str(d)
            
            if ((refreshCount> 0) and not(trialCount % refreshCount)):
                print_board_state(B, '.', True)
                print(f" Trial count : {trialCount}")
            #end

            if isValidSudoku(B):
                # S_previous = S.copy() # SolveM1
                
                try: 
                    S = squareOp(S,B,r,c)                 
                    # B = solve(B,S,dispBoard=True) # SolveM1
                    pass
                except:
                    # clearBdepth(depth) # SolveM1
                    return False
                #end
                
                
                solutionFound = backTrackSolve(depth+1)
                if solutionFound or not len(R):
                    return True
                else:
                    B[r,c] = 0;
                    # clearBdepth(depth) # SolveM1
                    # S = S_previous # SolveM1
                #end
            else:
                B[r,c] = 0;
            #end
        #end
    #end

    backTrackSolve(0)
    return (B,trialCount)
#end

## =========================================================================
## Backtrack generate via most constraining
def BacktrackMostConstrained(B, S, refreshCount=0, countLimit = 0, trialCount_BTG=0):
    if  countLimit > 0 and trialCount_BTG > countLimit:
        return (B, False, trialCount_BTG)

    def isValid(B,P):
        return not( P == 0 and B == 0 )

    if np.all(B > 0) and isValidSudoku(B):
        print_board_state(B, '.', True)
        print("Success")
        print(f" Trial count : {trialCount_BTG}")
        return (B, True , trialCount_BTG)
    
    if not isValidSudoku(B):
        return (B, False, trialCount_BTG)
    
    N = len(B)
    P = np.sum(S, axis=2) + (B>0)*(N+1);
    pmin = P.min()

    if np.any(P==0):
        return (B, False, trialCount_BTG)

    (rs,cs) = np.where( P == pmin) 
    if rs.size > 0:
        r = rs[0]
        c = cs[0]
        (vals,) = np.where(S[r,c,:])

        vals = list(vals)
        random.shuffle(vals) 

        for i in range(len(vals)):
            trialCount_BTG = trialCount_BTG+1;
            B_cp = B.copy()
            S_cp = S.copy()
            val = vals[i]+1# Adjust
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
            (B_Done, res, trialCount_BTG) = BacktrackMostConstrained(B_cp, S_cp, refreshCount=refreshCount, trialCount_BTG=trialCount_BTG)
            if res:
                return (B_Done , True , trialCount_BTG)
            #end
        #end
    #end

    return (B, False, trialCount_BTG)
#end
