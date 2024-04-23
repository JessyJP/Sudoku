import os
import numpy as np

## =========================================================================
# Display Functions
def print_board_state(board, substituteZero='', border=False, clear=True):
    N = len(board);
    g = int(np.sqrt(N))
    eS = len(str(abs(N)));  # Extra Space for digits
    if clear: os.system('cls' if os.name == 'nt' else 'clear')  # Clear the console screen

    border_line = '-' * ((eS + 1) * N + 2 * g + 1)  # is length of the border line
    for i, row in enumerate(board):
        if border and i % g == 0:
            print(border_line)
        # end
        print_row = []
        if border: print_row.append('|')  # end
        for j, item in enumerate(row):
            if item == 0 and substituteZero != '':
                print_row.append(substituteZero.rjust(eS))
            else:
                print_row.append(str(item).rjust(eS))
            # end
            if border and (j + 1) % g == 0 and (j + 1) != len(row):
                print_row.append('|')
            # end
        # end
        if border: print_row.append('|')  # end
        print(' '.join(print_row))
    # end
    if border:
        print(border_line)
    # end


# end

def print_solutionSpace(board, substituteZero='', border=False, clear=True):
    N = len(board);
    g = int(np.sqrt(N));
    eS = len(str(abs(N)));  # Extra Space for digits
    if clear: os.system('cls' if os.name == 'nt' else 'clear')  # end # Clear the console screen

    single_border_line = '-' * ((eS + 1) * N * g + 2 * (N + g) - 1)  # is length of the border line
    double_border_line = '=' * ((eS + 1) * N * g + 2 * (N + g) - 1)  # is length of the border line

    for outer_x, outer_row in enumerate(board):
        if border and outer_x % g == 0:
            print(double_border_line)
        # end
        for inner_x in range(g):
            print_row = []
            if border: print_row.append('|')  # end
            for outer_y, outer_col in enumerate(outer_row):
                for inner_y in range(g):
                    item = outer_col[inner_x * g + inner_y]
                    if item:
                        print_row.append(str(inner_x * g + inner_y + 1).rjust(eS))
                    else:
                        print_row.append(substituteZero.rjust(eS))
                    # end

                    if border and (inner_y + 1) % g == 0 and (inner_y + 1) < len(outer_col):
                        print_row.append('|')
                    # end
                # end

                if border and (outer_y + 1) % g == 0 and (outer_y + 1) < len(outer_row):
                    print_row.append('|')
                # end
            # end

            print(' '.join(print_row))
            if border and (inner_x + 1) % g == 0 and (inner_x + 1) < len(outer_row):
                print(single_border_line)
            # end
        # end
    # end

    if border:
        print(double_border_line)
    # end
# end

