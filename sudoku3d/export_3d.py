import numpy as np
import os
import subprocess
import glob
import webbrowser
from sudoku.export import compile_TEX_to_PDF, open_pdf, cleanup_auxiliary_files, open_html, compilePDF

# =========================================================================
# Export & Import solution to CSV
# =========================================================================


def load_sudoku_board_3d(filepath):
    """
    Load a 3D Sudoku board from a CSV file, where layers are separated by blank lines.
    """
    with open(filepath, 'r') as file:
        data = file.read().strip().split('\n\n')
    layers = [np.loadtxt(layer.strip().split('\n'), delimiter=",", dtype=int) for layer in data]
    return np.stack(layers, axis=-1)


def save_sudoku_board_3d(B, subdirectory="../Solutions", filename_prefix="Sudoku3DSolution"):
    """
    Save a 3D Sudoku board to a CSV file, separating layers with a blank line.
    """
    if not os.path.exists(subdirectory):
        os.makedirs(subdirectory)

    depth = B.shape[2]  # Depth of the 3D Sudoku grid
    ind = 1
    while True:
        filename = f"{filename_prefix}_{B.shape[0]}x{B.shape[1]}x{depth}_{ind}.csv"
        filepath = os.path.join(subdirectory, filename)
        if os.path.exists(filepath):
            existing_solution = load_sudoku_board_3d(filepath)
            if np.array_equal(existing_solution, B):
                print(f"A Sudoku solution with the same content already exists at: '{filepath}'.")
                return
            ind += 1
        else:
            break

    with open(filepath, 'w') as file:
        for z in range(depth):
            np.savetxt(file, B[:, :, z], fmt='%d', delimiter=',')
            if z < depth - 1:
                file.write('\n\n')
    print(f"Saved the 3D Sudoku solution to '{filepath}'.")


# ==========================================================================
# Export to HTML
# =========================================================================

def export_board_to_html_3d(B, filepath):
    """
    Export a 3D Sudoku board to an HTML file with tabs for each layer.
    """
    N, _, depth = B.shape
    html_content = '<!DOCTYPE html>\n<html lang="en">\n<head>\n<meta charset="UTF-8">\n'
    html_content += '<title>3D Sudoku Board</title>\n<style>\n'
    html_content += 'table {width: 50%; margin: auto; border-collapse: collapse;}'
    html_content += 'td {width: 40px; height: 40px; text-align: center; border: 1px solid #333;'
    html_content += 'font-family: Arial, sans-serif; font-size: 20px;}\n'
    html_content += '</style>\n</head>\n<body>\n'

    for z in range(depth):
        html_content += f'<h2>Layer {z + 1}</h2>\n<table>\n'
        for i in range(N):
            html_content += '<tr>\n'
            for j in range(N):
                value = B[i, j, z]
                html_content += f'<td>{value if value > 0 else ""}</td>\n'
            html_content += '</tr>\n'
        html_content += '</table>\n'

    html_content += '</body>\n</html>'

    with open(filepath, 'w') as f:
        f.write(html_content)
    print(f"Saved the 3D Sudoku board to '{filepath}'.")


# ==========================================================================
# Export to PDF
# =========================================================================

def export_board_to_TEX_3d(filepath, B):
    """
    Export a 3D Sudoku board to a LaTeX file using TikZ for drawing.
    """
    N, _, depth = B.shape
    unit = "cm"
    scale = 1
    squareSize = 1 * scale
    paperwidth, paperheight = (np.array(B.shape[0:2]) + 0.1) * squareSize
    # squareSize = squareSize * 0.98  # Scale to make just slightly smaller

    latex_content = '\n'.join([
        '\\documentclass[12pt]{extarticle}',
        '\\usepackage[paperwidth=' + str(paperwidth) + unit + ', paperheight=' + str(
            paperheight) + unit + ', margin=0' + unit + ']{geometry}',
        '\\usepackage{tikz}',
        '\\pagestyle{empty}',
        '\\begin{document}',
        '\\thispagestyle{empty}',
        '\\begin{center}'
    ])

    for z in range(depth):
        # latex_content += f'\\subsection*{{Layer {z + 1}}}'
        latex_content += '\\begin{tikzpicture}\n'
        latex_content += '\\draw[step=' + str(squareSize) + ', thin, gray] (0,0) grid (' + str(N) + ',' + str(
            N) + ');\n'

        for i in range(N):
            for j in range(N):
                val = B[i, j, z]
                if val > 0:
                    latex_content += f'\\node at ({i + 0.5},{j + 0.5}) {{{val}}};\n'

        latex_content += '\\end{tikzpicture}\n'

    latex_content += '\\end{center}\n\\end{document}'

    with open(filepath, 'w') as f:
        f.write(latex_content)
    print(f"Compiled the 3D Sudoku board to LaTeX at '{filepath}'.")


import numpy as np


def export_3d_board_to_TEX(filepath, B):
    """
    Export a 3D Sudoku board to a LaTeX file using TikZ for drawing each layer on a separate page.

    Parameters:
    - filepath: The path where the LaTeX file should be saved.
    - B: The 3D Sudoku board as a 3D numpy array.
    """
    Z, N, _ = B.shape  # Assuming B is a cubic 3D array, ZxNxN
    g = int(np.sqrt(N))  # Size of the block, assuming square root of N is an integer
    unit = "cm"
    scale = 1
    squareSize = 1 * scale
    paperwidth, paperheight = (N + 1) * squareSize, (N + 1) * squareSize

    def squareEntry(r, c, val):
        r = r + 0.5
        c = c + 0.5
        return rf"\node at ({c}\SquareSize, {r}\SquareSize) {{${val}$}};" + '\n\t'

    latex_content = rf'''
\documentclass[12pt]{{extarticle}}
\usepackage[paperwidth={paperwidth}{unit}, paperheight={paperheight}{unit}, margin=1{unit}]{{geometry}}
\usepackage{{tikz}}
\begin{{document}}
\newlength{{\SquareSize}}
\setlength{{\SquareSize}}{{{squareSize}{unit}}}
'''

    for z in range(Z):
        allEntries = []
        for r in range(N):
            for c in range(N):
                val = B[z, r, c]
                if val > 0:
                    allEntries.append(squareEntry(r, c, val))

        allEntriesStr = ''.join(allEntries)

        latex_content += rf'''
\newpage
\begin{{center}}
\begin{{tikzpicture}}
    \draw[step=\SquareSize, gray!50, thin] (0,0) grid ({N}\SquareSize,{N}\SquareSize);
    \draw[step={g}\SquareSize, black, very thick] (0,0) grid ({N}\SquareSize,{N}\SquareSize);
    \draw[black, ultra thick] (0,0) rectangle ({N}\SquareSize,{N}\SquareSize);
    {allEntriesStr}
\end{{tikzpicture}}
\end{{center}}
'''

    latex_content += r'''
\end{document}
    '''
    with open(filepath, 'w') as f:
        f.write(latex_content)
    print(f"Saved the 3D Sudoku board to '{filepath}' as a multi-page LaTeX document.")




