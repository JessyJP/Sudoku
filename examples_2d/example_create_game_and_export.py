from sudoku import *
from sudoku.make_game import select_board_file, make_game_from_solution

B = load_sudoku_board(select_board_file("../Solutions/", index=None));


print_board_state(B, ".", border=True, clear=True)
B_least = B.copy()
for t in range(300):# try 10 times
    try:
        B_current = make_game_from_solution(B.copy())
    except:
        continue
    currentSum = sum(sum(B_current>0))
    newSum = sum( sum(B_least>0))
    if currentSum < newSum:
        print(f"TEST {t}: Current number count {currentSum} < Smallest number count {newSum}")
        B_least = B_current
B = B_least

print_board_state(B, ".", border=True, clear=False)
# save_sudoku_board(B,"Games")

directory = r"r:/"
filename = r"sudoku"
fileExt = r".tex"
export_board_to_TEX(directory+filename+fileExt,B)

compilePDF(directory,filename+fileExt)

html_file = directory+filename+".html"
export_board_to_html(B, html_file)

open_html(html_file)
open_pdf(directory+filename+r".pdf")
