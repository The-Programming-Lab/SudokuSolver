from sudokuRules import Checker
from sudoku import Sudoku
import numpy as np

# https://en.wikipedia.org/wiki/Exact_cover#Sudoku

# Followed this Java implementation
# https://www.baeldung.com/java-sudoku

# Sudoku is an exact cover solution. To convert the representation, we look at the puzzle as a binary matrix size 729 x 324.

# 729: This is the total possible choices we can make. 9 rows x 9 cols = 81 cells total. Each cell can be 9 digits. cell R1C1 can be {1,2,3,4,5,6,7,8,9}, R1C2 can be {1,2,3,4,5,6,7,8,9} so forth.
# 324: This is the sets of constraints on the puzzle we have. There are 4 types of constraints:
    # 1) Row-Column: Essentailly a cell. Each cell is bound by only having one number allowed.
        # - there is 9 rows and 9 columns = 81 total constraint sets

    # 2) Row-Number: In one row of the puzzle, there are 9 columns. For every num (1-9), it can fit in either of the 9 columns per row.
        # Example for Row 1 and the Number 1 = {R1C1,#1, R1C2#1, R1C3#1, R1C4#1, R1C5#1, R1C6#1, R1C7#1, R1C8#1, R1C9#1}
        # - there is 9 rows and 9 numbers = 81 total constraint sets

    # 3) Column-Number: Inversely, in one column of the puzzle, there are 9 rows. For every num (1-9), it can fit in either of the 9 rows per column.
        # Example for Column 1 and the Number 1 = {R1C1,#1, R2C1#1, R3C1#1, R4C1#1, R5C1#1, R6C1#1, R7C1#1, R8C1#1,R9C1#1}
        # - there is 9 rows and 9 numbers = 81 total constraint sets

    # 4) Box-Number: There are 9 boxes within the puzzle, each 3x3 in size. For every num (1-9) it can fit in one of the 3x3 cells per box
        # Example for Box 1 and the Number 1 = {R1C1#1, R1C2#1, R1C3#1, R2C1#1, R2C2#1, R2C3#1, R3C1#1, R3C2#1, R3C3#1}
        # - there is 9 rows and 9 numbers = 81 total constraint sets

    # In total 81 * 4 types of constraints = 324 constraint sets

int BOARD_SIZE = 9
int BOX_SIZE = 3
int NO_VALUE = 0
int CONSTRAINTS = 4
int MIN_VALUE = 1
int MAX_VALUE = 9
int COVER_START_INDEX = 1

# Function to index the exact row we are in
def getIndex(row, column, value):
    return (row-1) * BOARD_SIZE * BOARD_SIZE + (column-1) * BOARD_SIZE + (value-1)

def createExactCover():
    # Create binary matrix with 729 rows and 324 columns
    matrix = np.zeros(BOARD_SIZE * BOARD_SIZE * MAX_VALUE, BOARD_SIZE * BOARD_SIZE * CONSTRAINTS)

    # Index of which column in the constraints we are in
    constraintCol = 0

# constraintCol represents starting index for this section of constraints
# Loop through every possible cell in puzzle[row, col] and every possible value to mark the correct cell in exact cover matrix
def checkCellConstraint(exactCoverBoard, constraintCol):
    # Iterating through the possible rows (1-9)
    for r in range(COVER_START_INDEX, BOARD_SIZE):
        # Iterating through the possible cols (1-9)
        for c in range(COVER_START_INDEX, BOARD_SIZE):
            # Possible values for each cell is (1-9). 
            for n in range(MIN_VALUE, MAX_VALUE+1):
                # Using the row, col and value, we index which exact cover board row needs to be marked for constraintCol
                    # For ex: constraintCol = 0 (R1C1) is going to be marked for rows R1C1#1 - R1C1#9
                index = getIndex(r,c,n)
                exactCoverBoard[index][constraintCol] = 1
        # Move onto next constraintCol
        constraintCol += 1
    # Return where we left off marking the constraintColumns
    return constraintCol

# constraintCol represents starting index for this section of constraints
# Loop through every possible row in the puzzle and every possible value to mark correct cell in exact cover matrix
def checkRowConstraint(exactCoverBoard, constraintCol):
    # Iterating through the possible rows (1-9)
    for r in range(COVER_START_INDEX, BOARD_SIZE):
        # Iterating through the possible values in each row (1-9)
        for n in range(MIN_VALUE, MAX_VALUE+1):
            # Possible columns each value in a row could be in
            for c in range(COVER_START_INDEX, BOARD_SIZE):
                # Using the row, col and value, we index which exact cover board row needs to be marked for constraintCol
                    # For ex: constraintCol = 81 (R1#1) is going to be marked for rows R1C1#1 - R1C9#1
                index = getIndex(r,c,n)
                exactCoverBoard[index][constraintCol] = 1
        constraintCol += 1
    # Return where we left off marking the constraintColumns
    return constraintCol

# constraintCol represents starting index for this section of constraints
# Loop through every possible column in the puzzle and every possible value to mark correct cell in exact cover matrix
def checkColumnConstraint(exactCoverBoard, constraintCol):
    # Iterating through the possible columns (1-9)
    for c in range(COVER_START_INDEX, BOARD_SIZE):
        # Iterating through the possible values in each columnm (1-9)
        for n in range(MIN_VALUE, MAX_VALUE+1):
            # Possible rows each value in a column could be in
            for r in range(COVER_START_INDEX, BOARD_SIZE):
                # Using the row, col and value, we index which exact cover board row needs to be marked for constraintCol
                    # For ex: constraintCol = 162 (C1#1) is going to be marked for rows R1C1#1 - R9C1#1
                index = getIndex(r,c,n)
                exactCoverBoard[index][constraintCol] = 1
        constraintCol += 1
    # Return where we left off marking the constraintColumns
    return constraintCol
    
# constraintCol represents starting index for this section of constraints
# Loop through every possible block in the puzzle (there are 9 )and every possible value to mark correct cell in exact cover matrix
def checkBlockConstraint(exactCoverBoard, constraintCol):
    # Iterating through the possible columns (1-9)
    for c in range(COVER_START_INDEX, BOARD_SIZE):
        # Iterating through the possible values in each columnm (1-9)
        for n in range(MIN_VALUE, MAX_VALUE+1):
            # Possible rows each value in a column could be in
            for r in range(COVER_START_INDEX, BOARD_SIZE):
                # Using the row, col and value, we index which exact cover board row needs to be marked for constraintCol
                    # For ex: constraintCol = 162 (C1#1) is going to be marked for rows R1C1#1 - R9C1#1
                index = getIndex(r,c,n)
                exactCoverBoard[index][constraintCol] = 1
        constraintCol += 1
    # Return where we left off marking the constraintColumns
    return constraintCol


if __name__ == '__main__':
    # difficulty = 0.5 means 50% of spaces are blank
    puzzle = [
    [0,0,7,0,4,0,0,0,0],
    [0,0,0,0,0,8,0,0,6],
    [0,4,1,0,0,0,9,0,0],
    [0,0,0,0,0,0,1,7,0],
    [0,0,0,0,0,6,0,0,0],
    [0,0,8,7,0,0,2,0,0],
    [3,0,0,0,0,0,0,0,0],
    [0,0,0,1,2,0,0,0,0],
    [8,6,0,0,7,6,0,0,5]
    ]
