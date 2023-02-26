
import time
import pandas as pd
import numpy as np
from sudokuRules import Checker

# brute force sudoku solver
class Solver():
    checker = Checker()
    boardSize = 9
    boxSize = 3
    
    def getFirstZero(self, board):
        for i in range(self.boardSize):
            for j in range(self.boardSize):
                if board[i,j] == 0:
                    return (i,j)
        #raise("No zeros found")
        return None
   
    def solve_brute(self, board):
        stack = []

        while self.checker.checkSolved(board) == False:
            # place 1 in all empty cells and add them to the stack
            for i in range(self.boardSize):
                for j in range(self.boardSize):
                    if board[i,j] == 0:
                        board[i,j] = 1
                        stack.append((i,j))

            if self.checker.checkSolved(board) == True:
                return board
            
            # increment the last cell in the stack that is not 9
            # if the cell is 9, reset it to 0 and pop it from the stack
            firstZero = stack.pop()
            while(board[firstZero] == 9):
                board[firstZero] = 0
                firstZero = stack.pop()
            
            board[firstZero] += 1
            stack.append(firstZero)
        
        return board
            
    def solve_backtracking(self, board):
        # find the first empty cell
        for i in range(self.boardSize):
            for j in range(self.boardSize):
                if board[i,j] == 0:
                    # try all possible numbers
                    for k in range(1,10):
                        board[i,j] = k
                        if self.checker.checkAll(board):
                            # if the board is valid, solve the rest of the board
                            if self.solve_backtracking(board):
                                return True
                    # if no number is valid, reset the cell and return False
                    board[i,j] = 0
                    return False
        # if there are no empty cells, the board is solved
        return True
    


if __name__ == '__main__':
    data = pd.read_csv('sudoku.csv', header=None, nrows=10)
    # get the second row of the dataframe
    board = data.iloc[1]
    # covert the board string to a list of integers1
    puzzle = np.array(list(map(int, board[0]))).reshape(9,9)
    solution = np.array(list(map(int, board[1]))).reshape(9,9)
    '''
    [6 7 9 5 1 8 2 4 3]
    [5 4 3 7 2 9 6 1 8]
    [8 2 1 6 3 4 9 5 7]
    [7 9 4 3 5 2 1 8 6]
    [3 5 8 4 6 1 7 2 9]
    [2 1 6 8 9 7 5 3 4]
    [4 8 5 2 7 6 3 9 1]
    [9 6 2 1 8 3 4 7 0]
    [1 3 7 9 4 5 8 6 2] 

    [0 7 0 0 0 0 0 4 3]
    [0 4 0 0 0 9 6 1 0]
    [8 0 0 6 3 4 9 0 0]
    [0 9 4 0 5 2 0 0 0]
    [3 5 8 4 6 0 0 2 0]
    [0 0 0 8 0 0 5 3 0]
    [0 8 0 0 7 0 0 9 1]
    [9 0 2 1 0 0 0 0 5]
    [0 0 7 0 4 0 8 0 2]
    '''
    puzzle = np.array([
            [0, 7, 9, 5, 1, 8, 2, 4, 3],
            [5, 4, 3, 7, 2, 9, 6, 1, 8],
            [8, 2, 1, 6, 3, 4, 9, 5, 7],
            [7, 9, 4, 3, 5, 2, 1, 8, 6],
            [3, 5, 8, 4, 6, 1, 7, 2, 9],
            [2, 1, 6, 8, 9, 7, 5, 3, 4],
            [4, 8, 5, 2, 7, 6, 3, 9, 1],
            [9, 6, 2, 1, 8, 3, 4, 7, 5],
            [1, 3, 7, 9, 4, 5, 8, 6, 2]])

    print(puzzle)
    solver = Solver()

    # get time
    start = time.time()
    print(solver.solve_brute(puzzle))
    end = time.time()
    print(end - start)
    #print(puzzle)
    print(np.array_equal(puzzle, solution))
