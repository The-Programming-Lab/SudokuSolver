import numpy as np
import pandas as pd


class Checker():
    boardSize = 9
    boxSize = 3
    
    def checkRow(self, row):
        return self.checkList(self.board[row])

    def checkCol(self, col):
        return self.checkList(self.board[:,col])
    
    def checkBox(self, box):
        row = (box // self.boxSize) * self.boxSize
        col = (box % self.boxSize) * self.boxSize
        return self.checkList(self.board[row:row+self.boxSize,col:col+self.boxSize].flatten())

    def checkList(self, lst):
        # remove all zeros from the list
        lst = [i for i in lst if i != 0]
        # check if the list has duplicates
        # or if the list has numbers outside the range 1-9
        return len(lst) == len(set(lst)) and min(lst) >= 1 and max(lst) <= 9

    def checkAll(self, board, printErrors=False):
        self.board = board
        for i in range(self.boardSize):
            if not self.checkRow(i):
                if printErrors:
                    print("row", i, "failed")
                return False
            if not self.checkCol(i):
                if printErrors:
                    print("col", i, "failed")
                return False
            if not self.checkBox(i):
                if printErrors:
                    print("box", i, "failed")
                return False
        return True
    
    def checkSolved(self, board):
        return self.checkAll(board) and 0 not in board

    def findNextEmptyCell(self, board):
        for i in range(self.boardSize):
            for j in range(self.boardSize):
                if board[i][j] == 0:
                    return [i,j]

# main
if __name__ == '__main__':
    NUMBER = 100_0
    data = pd.read_csv('sudoku.csv', header=None, nrows=NUMBER)

    count = 0
    checker = Checker()
    for i in range(1,NUMBER):
        board = data.iloc[i]
        #puzzle = list(map(int, board[0]))
        solution = list(map(int, board[1]))
        solution = np.array(solution).reshape(9,9)

        if not checker.checkAll(solution):
            print("Solution should be valid")
            continue

        solution[1][2] += 1
        
        if checker.checkAll(solution):
            print("Solution should be invalid")
            continue

    # check for negative numbers
    board = data.iloc[1]
    solution = list(map(int, board[1]))
    solution = np.array(solution).reshape(9,9)
    solution[1][2] = -1
    if checker.checkAll(solution):
            print("Solution should be invalid")

    board = data.iloc[2]
    solution = list(map(int, board[1]))
    solution = np.array(solution).reshape(9,9)
    solution[8][8] = -5
    if checker.checkAll(solution):
            print("Solution should be invalid")

