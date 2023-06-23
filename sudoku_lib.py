import os;
import subprocess
import glob
import time
import numpy as np
import random
from indexing_lib import *

## =========================================================================

def InitializeBoard( N ):
    return  np.full((N, N), 0)
#end

## Solution Initialization function
def InitializeSolutionSpace(B,dispSolutionState=False):#
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
        if dispBoard: print_boardState(B,'.',True) #end
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
    (R, C) = get_Linear_Element_indices_ofNonZero(B)

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
        S = InitializeSolutionSpace(B);
        possibleValues = range(1, N+1)
        possibleValues = np.where(S[r,c,:])[0]+1
        randomPermutation = list(possibleValues)
        random.shuffle(randomPermutation) 
        for d in randomPermutation:
            trialCount = trialCount+1;

            B[r,c] = str(d)
            
            if not(trialCount % refreshCount):
                print_boardState(B,'.',True)
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
    (R, C) = get_Linear_Element_indices_ofNonZero(B)

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
            print_boardState(B,'.',True)
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
        S = InitializeSolutionSpace(B);
        possibleValues = range(1, N+1)
        possibleValues = np.where(S[r,c,:])[0]+1
        randomPermutation = list(possibleValues)
        # random.shuffle(randomPermutation) 
        for d in randomPermutation:
            trialCount = trialCount+1;

            B[r,c] = str(d)
            
            if ((refreshCount> 0) and not(trialCount % refreshCount)):
                print_boardState(B,'.',True)
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
global trialCount_BTG
trialCount_BTG = 0
def BacktrackMostConstrained(B, S, refreshCount=0):
    def isValid(B,P):
        return not( P == 0 and B == 0 )

    global trialCount_BTG
    if np.all(B > 0) and isValidSudoku(B):
        print_boardState(B,'.',True)
        print("Success")
        print(f" Trial count : {trialCount_BTG}")
        return (True, B)
    if not isValidSudoku(B):
        return (False, B)
    
    N = len(B)
    P = np.sum(S, axis=2) + (B>0)*(N+1);
    pmin = P.min()

    if np.any(P==0):
        return (False, B)

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
                print_boardState(B,'.',True)
                print(f" Trial count : {trialCount_BTG}")
            #end
            # solve(B,S,dispBoard=True)
            (res,B_Done) = BacktrackMostConstrained(B_cp, S_cp, 2+int(np.log2(trialCount_BTG)))
            if res:
                return (True,B_Done)
            #end
        #end
    #end

    return (False,B)
#end

## =========================================================================
## Additional check functions
# Check board fullness
def count_non_zero(matrix):
    return sum(1 for row in matrix for item in row if item != 0)
#end

# Solution check
def compare_boards(Board, solution):
    if np.array_equal(Board, solution):
        print("The boards are identical!!! SOLUTION FOUND!")
    else:
        print("The boards are !!=///==!! not identical.")
    #end
#end

# Conversion function
def convert_to_int_board(board):
    intBoard = [[int(cell) if cell != "." else 0 for cell in row] for row in board]
    intBoard = np.array(intBoard)
    return intBoard
#end

## =========================================================================
# Display Functions 
def print_boardState(board, substituteZero='', border=False,clear=True):
    N = len(board); g = int(np.sqrt(N))
    eS = len(str(abs(N)));# Extra Space for digits
    if clear: os.system('cls' if os.name == 'nt' else 'clear')  # Clear the console screen

    border_line = '-' * ((eS+1)*N + 2*g +1) # is length of the border line
    for i, row in enumerate(board):
        if border and i % g == 0:
            print(border_line)
        #end
        print_row = []
        if border: print_row.append('|') #end
        for j, item in enumerate(row):
            if item == 0 and substituteZero != '':
                print_row.append(substituteZero.rjust(eS))
            else:
                print_row.append(str(item).rjust(eS))
            #end
            if border and (j + 1) % g == 0 and (j + 1) != len(row):
                print_row.append('|')
            #end
        #end
        if border: print_row.append('|') #end
        print(' '.join(print_row))
    #end
    if border:
        print(border_line)
    #end
#end

def print_solutionSpace(board, substituteZero='', border=False, clear=True):
    N = len(board); g = int(np.sqrt(N));
    eS = len(str(abs(N)));# Extra Space for digits
    if clear: os.system('cls' if os.name == 'nt' else 'clear') #end # Clear the console screen
    
    single_border_line = '-' * ((eS+1)*N*g + 2*(N + g)-1) # is length of the border line
    double_border_line = '=' * ((eS+1)*N*g + 2*(N + g)-1) # is length of the border line
    
    for outer_x, outer_row in enumerate(board):
        if border and outer_x % g == 0:
            print(double_border_line)
        #end
        for inner_x in range(g):
            print_row = []
            if border: print_row.append('|') #end
            for outer_y, outer_col in enumerate(outer_row):
                for inner_y in range(g):
                    item = outer_col[inner_x * g + inner_y]
                    if item:
                        print_row.append(str(inner_x * g + inner_y + 1).rjust(eS))
                    else:
                        print_row.append(substituteZero.rjust(eS))
                    #end

                    if border and (inner_y + 1) % g == 0 and (inner_y + 1) < len(outer_col):
                        print_row.append('|')
                    #end
                #end

                if border and (outer_y + 1) % g == 0 and (outer_y + 1) < len(outer_row):
                    print_row.append('|')
                #end
            #end

            print(' '.join(print_row))
            if border and (inner_x + 1) % g == 0 and (inner_x + 1) < len(outer_row):
                print(single_border_line)
            #end
        #end
    #end

    if border:
        print(double_border_line)
    #end
#end

## =========================================================================
# Export & Import solution

def load_sudoku_board(filepath):
    # Load the solution from a CSV file into a NumPy array
    solution = np.loadtxt(filepath, delimiter=",", dtype=int)
    solution = np.array(solution)
    return solution
#end


def save_sudoku_board(B, subdirectory="Solutions"):
    if not os.path.exists(subdirectory):
        os.makedirs(subdirectory)
    #end
    
    N = B.shape[0] # Get the size of the Sudoku matrix

    # Generate the file names and check if they already exist and the content is the same
    ind = 1
    while True:
        filename = f"SudokuSolution_{N}x{N}_{ind}.csv"
        filepath = os.path.join(subdirectory, filename)
        if os.path.exists(filepath):
            # Load the existing solution and compare it to the new one
            existing_solution = load_sudoku_board(filepath)
            if np.array_equal(existing_solution.squeeze(), B.squeeze()):
                print(f" ==> A Sudoku solution with the same content already exists at : '{filepath}'.")
                return
            #end
            ind += 1
        else:
            break
        #end
    #end

    # Save the array to a CSV file
    np.savetxt(filepath, B, delimiter=",", fmt="%d")

    print(f"Saved the Sudoku solution of size {N}x{N} to '{filepath}'.")
#end

## ==========================================================================
# Export to PDF

def export_board_to_TEX(filepath, B):
    N = len(B)
    g = int(np.sqrt(N))
    unit = "cm"
    scale = 1
    squareSize = 1*scale
    paperwidth, paperheight = (np.array(B.shape) + 0.1 + g*0)*squareSize

    def squareEntry(r, c, val):
        r = r+0.5
        c = c+0.5
        texLine = rf"\node at ({r}\SquareSize, {c}\SquareSize) {{${val}$}};"+'\n\t'
        return texLine
    #end

    allEntries = []
    for r in range(N):
        for c in range(N):
            val = B[r, c]
            if val > 0:
                allEntries.append(squareEntry(r, c, val))
            #end
        #end
    #end
    allEntriesStr = ''.join(allEntries)

    latexTemplate = fr'''
\documentclass[12pt]{{extarticle}}
\usepackage[paperwidth={paperwidth}{unit}, paperheight={paperheight}{unit}, 
            %lmargin={0}{unit}, rmargin={0}{unit}, 
            %tmargin={0}{unit}, 
            % bmargin={1}{unit},
             margin={0}{unit}
        ]{{geometry}}
\usepackage{{tikz}}
%\usepackage{{adjustbox}}

\newlength{{\SquareSize}}
\setlength{{\SquareSize}}{{{squareSize}{unit}}}

\newcommand{{\SudokuGrid}}{{
\begin{{tikzpicture}}%[scale=0.99][t!]
    \draw[step=\SquareSize,gray!50,thin] (0,0) grid ({N}\SquareSize,{N}\SquareSize);
    \draw[step={g}\SquareSize,black,very thick] (0,0) grid ({N}\SquareSize,{N}\SquareSize);
    \draw[black,ultra thick] (0,0) rectangle ({N}\SquareSize,{N}\SquareSize);
    % All number entries
    {allEntriesStr}
\end{{tikzpicture}}
}}

\begin{{document}}
\thispagestyle{{empty}}
%\vspace*{{\fill}}
\begin{{center}}
%\begin{{adjustbox}}{{max width=\textwidth, max height=\textheight, keepaspectratio}}
    \SudokuGrid
\thispagestyle{{empty}}
\end{{center}}
%\vspace*{{\fill}}
\end{{document}}
    '''

    with open(filepath, 'w') as f:
        f.write(latexTemplate)
    #end
#end

def compilePDF(directory, filename):
    try:
        subprocess.check_call(["pdflatex", f"-output-directory={directory}", directory + filename], shell=True)
    except subprocess.CalledProcessError:
        print("Error: Compilation failed")
    else:
        # If pdflatex ran successfully, delete the auxiliary files
        base_filename = os.path.splitext(filename)[0]  # Get the base filename (without extension)
        aux_files = glob.glob(directory + base_filename + '.*')  # List all files with the same base filename
        for file in aux_files:
            if file.endswith('.log') or file.endswith('.aux'):
                os.remove(file)
                print(f"Auxiliary file {file} was deleted!")
            #end
        #end
    #end
#end

def open_pdf(file):
    try:
        os.system(f' {file}')  # On Windows
        # os.system(f'xdg-open {file}')  # On Linux
        # os.system(f'open {file}')   # On Mac
        print(f"Open: {file}")
    except Exception as e:
        print(f'Error: {e}')
    #end
#end