import numpy as np
import os
import subprocess
import glob
import webbrowser


# =========================================================================
# Export & Import solution to CSV
# =========================================================================


def load_sudoku_board(filepath):
    """
    Load a Sudoku board from a CSV file.

    Parameters:
    - filepath: The path to the CSV file containing the Sudoku board.

    Returns:
    - The loaded Sudoku board as a 2D numpy array.
    """
    return np.loadtxt(filepath, delimiter=",", dtype=int)


def save_sudoku_board(B, subdirectory="../Solutions", filename_prefix="SudokuSolution"):
    """
    Save a Sudoku board to a CSV file.

    Parameters:
    - B: The Sudoku board as a 2D numpy array.
    - subdirectory: The subdirectory under which to save the file.
    - filename_prefix: The prefix for the generated filename.

    Ensures that the file is uniquely named if a file with the same content exists.
    """
    if not os.path.exists(subdirectory):
        os.makedirs(subdirectory)

    N = B.shape[0]  # Get the size of the Sudoku matrix

    # Attempt to save the file with an increasing index to avoid overwriting
    ind = 1
    while True:
        filename = f"{filename_prefix}_{N}x{N}_{ind}.csv"
        filepath = os.path.join(subdirectory, filename)
        if os.path.exists(filepath):
            # Load the existing solution and compare it to the new one
            existing_solution = load_sudoku_board(filepath)
            if np.array_equal(existing_solution, B):
                print(f"A Sudoku solution with the same content already exists at: '{filepath}'.")
                return
            ind += 1
        else:
            break

    # Save the array to a CSV file
    np.savetxt(filepath, B, delimiter=",", fmt="%d")
    print(f"Saved the Sudoku solution of size {N}x{N} to '{filepath}'.")


# ==========================================================================
# Export to HTML
# =========================================================================

def export_board_to_html(B, filepath):
    """
    Export a Sudoku board to an HTML file with enhanced visual distinction for 3x3 blocks.

    Parameters:
    - B: The Sudoku board as a 2D numpy array.
    - filepath: The path where the HTML file should be saved.
    """
    N = len(B)
    g = int(np.sqrt(N))  # Determine the size of each block
    html_content = '<!DOCTYPE html>\n<html lang="en">\n<head>\n<meta charset="UTF-8">\n'
    html_content += '<title>Sudoku Board</title>\n<style>\n'
    html_content += 'table {width: 50%; margin: auto; border-collapse: collapse;}'
    html_content += 'td {width: 40px; height: 40px; text-align: center; border: 1px solid #333;'
    html_content += 'font-family: Arial, sans-serif; font-size: 20px;}'
    html_content += 'td.block-border-right {border-right: 3px solid black;}'
    html_content += 'td.block-border-bottom {border-bottom: 3px solid black;}'
    html_content += '</style>\n</head>\n<body>\n'
    html_content += '<table>\n'

    for i, row in enumerate(B):
        html_content += '<tr>\n'
        for j, value in enumerate(row):
            # Add special classes for cells that are on the right or bottom edge of a block
            classes = []
            if (j + 1) % g == 0 and j < N - 1:
                classes.append("block-border-right")
            if (i + 1) % g == 0 and i < N - 1:
                classes.append("block-border-bottom")
            class_str = ' '.join(classes)
            class_attr = f' class="{class_str}"' if classes else ''
            html_content += f'<td{class_attr}>{value if value > 0 else ""}</td>\n'
        html_content += '</tr>\n'
    html_content += '</table>\n</body>\n</html>'

    with open(filepath, 'w') as f:
        f.write(html_content)
    print(f"Saved the Sudoku board to '{filepath}'.")


def open_html(filepath):
    """
    Open an HTML file in the default web browser.

    Parameters:
    - filepath: The path to the HTML file.
    """
    # Convert the filepath to an absolute path
    absolute_path = os.path.abspath(filepath)

    # Check if the file exists before trying to open it
    if os.path.exists(absolute_path):
        # Open the HTML file in the default browser
        webbrowser.open('file://' + absolute_path, new=2)  # new=2 specifies that the file should open in a new tab.
        print(f"Opened '{filepath}' in the default browser.")
    else:
        print(f"Error: The file '{filepath}' does not exist.")


# ==========================================================================
# Export to PDF
# =========================================================================


def export_board_to_TEX(filepath, B):
    """
    Export a Sudoku board to a LaTeX file using TikZ for drawing.

    Parameters:
    - filepath: The path where the LaTeX file should be saved.
    - B: The Sudoku board as a 2D numpy array.
    """
    N = len(B)
    g = int(np.sqrt(N))
    unit = "cm"
    scale = 1
    squareSize = 1 * scale
    paperwidth, paperheight = (np.array(B.shape) + 0.1 + g * 0) * squareSize

    def squareEntry(r, c, val):
        r = r + 0.5
        c = c + 0.5
        texLine = rf"\node at ({r}\SquareSize, {c}\SquareSize) {{${val}$}};" + '\n\t'
        return texLine

    allEntries = []
    for r in range(N):
        for c in range(N):
            val = B[r, c]
            if val > 0:
                allEntries.append(squareEntry(r, c, val))

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


def compile_TEX_to_PDF(tex_filepath):
    """
    Compile a LaTeX file to a PDF using pdflatex.

    Parameters:
    - tex_filepath: The path to the LaTeX file.
    """
    directory, filename = os.path.split(tex_filepath)
    try:
        subprocess.run(["pdflatex", f"-output-directory={directory}", filename], check=True, cwd=directory)
        print(f"Compiled '{tex_filepath}' to PDF successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Error compiling '{tex_filepath}' to PDF: {e}")


def open_pdf(file):
    try:
        os.system(f' {file}')  # On Windows
        # os.system(f'xdg-open {file}')  # On Linux
        # os.system(f'open {file}')   # On Mac
        print(f"Open: {file}")
    except Exception as e:
        print(f'Error: {e}')


def cleanup_auxiliary_files(directory, base_filename):
    """
    Remove auxiliary files generated by LaTeX compilation.

    Parameters:
    - directory: The directory containing the files.
    - base_filename: The base filename without extension.
    """
    for ext in ['.log', '.aux', '.out']:
        filepath = os.path.join(directory, f"{base_filename}{ext}")
        if os.path.exists(filepath):
            os.remove(filepath)
            print(f"Deleted auxiliary file: {filepath}")
