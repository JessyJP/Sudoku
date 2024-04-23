from sudoku import *



def get_filepath(directory, index=None):
    files = os.listdir(directory) # Get a list of files in the directory

    # If the index is not provided or is out of range, pick a random file
    if index is None or index >= len(files):
        return os.path.join(directory, random.choice(files))
    #end

    # If the index is valid, return the corresponding file
    return os.path.join(directory, files[index])
    print(f"File selected ""{files[index]}""  in directory:""{directory}""")
#end

B = load_sudoku_board(get_filepath("./Solutions/",index=None));



def makeGameFromSolution(B_full,display=False):
    # Get the total size
    size = B_full.size

    # Get the indices of the elements in the matrix
    row_indices, col_indices = np.where(B_full)

    # Generate a permutation of the indices
    permutation = np.random.permutation(size)

    # Shuffle the row and column indices
    Rs = row_indices[permutation]
    Cs = col_indices[permutation]

    def removeElements(B, n):
        # Set the first n elements corresponding to the shuffled indices to 0
        for i in range(n):
            B[Rs[i], Cs[i]] = 0
        #end
        return B
    #end

    # Interval bisection
    upper = size;
    lower = 0;
    while (upper != lower and abs(upper-lower)>1):
        n = int((upper+lower)/2)
        # Remove elements from B
        B = removeElements(B_full.copy(), n)
        # print_boardState(B,substituteZero=".",border=True)
        if display: print(f"\n Test Remove n: {n} with DEL Upper Limit: {upper} & DEL Lower Limit: {lower}")
        S = InitializeSolutionSpace(B)
        B = solve(B,S,dispBoard=display)

        if np.array_equal(B,B_full):
            lower = n
        else:
            upper = n
        #end
    #end

    n = upper
    B = removeElements(B_full, n)
    if display:
        print_boardState(B,substituteZero=".",border=True)
        print(f"Current number count {sum(sum(B>0))}")
    return B
#end


print_boardState(B,".",border=True,clear=True)
B_least = B.copy()
for t in range(300):# try 10 times
    B_current = makeGameFromSolution(B.copy())
    currentSum = sum(sum(B_current>0))
    newSum = sum( sum(B_least>0))
    if currentSum < newSum:
        print(f"TEST {t}: Current number count {currentSum} < Smallest number count {newSum}")
        B_least = B_current
B = B_least

print_boardState(B,".",border=True,clear=False)
# save_sudoku_board(B,"Games")

directory = r"r:/"
filename = r"sudoku"
fileExt = r".tex"
export_board_to_TEX(directory+filename+fileExt,B)

compilePDF(directory+filename+fileExt)

open_pdf(directory+filename+r".pdf")

