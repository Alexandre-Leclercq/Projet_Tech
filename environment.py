# -*- coding: utf-8 -*-
"""
Created on Wed Dec 21 10:45:30 2022

@author: Sébastien CIVADE, Adrien COLMART, Thomas FOY, Alexandre LECLERCQ
"""
import random
import torch
from typing import Optional


class SimpleMaze:

    ACTIONS: dict = {
        "north": (-1, 0),
        "east": (0, 1),
        "south": (1, 0),
        "west": (0, -1)
    }

    def __init__(self, row: int, col: int, seed: int = 0):
        self.__row = row
        self.__col = col
        self.__seed: int = seed
        self.start_point: tuple = [row - 1, 0]
        self.character_pos: list = self.start_point[:].copy()
        self.end_point: tuple = [0, col - 1]
        #  self.reset(seed)

    def reset(self, seed: Optional[int] = None) -> list:
        self.__seed = (seed, self.__seed)[seed is None]
        random.seed(self.__seed)
        self.character_pos: list[int, int] = self.start_point[:]
        return self.character_pos.copy()

    def done(self) -> bool:
        return self.character_pos == self.end_point

    def reward(self) -> float:
        if self.character_pos == self.end_point:
            return 1000
        else:
            return -1

    def step(self, action: int) -> (list, int, bool):
        movement = self.ACTIONS[action]
        if self.__row > self.character_pos[0] + movement[0] >= 0 and self.__col > self.character_pos[1] + movement[1] >= 0:
            self.character_pos[0] += movement[0]
            self.character_pos[1] += movement[1]
        return self.character_pos.copy(), self.reward(), self.done()

    """    define the actions doable    """

    def render(self, mode: str = "computed") -> None:
        if mode == "computed":
            for i in range(self.__row):
                print(i, end="\t")
                for j in range(self.__col):
                    print("|", end="")
                    if self.character_pos == [i, j]:
                        print("C", end="")
                    elif self.end_point == [i, j]:
                        print("E", end="")
                    else:
                        print(".", end="")
                print("|")
            print("")
        elif mode == "human":
            print("human")


class Maze:
    ACTIONS: dict = {
        "north": (-1, 0),
        "east": (0, 1),
        "south": (1, 0),
        "west": (0, -1)
    }

    def __init__(self, row: int, col: int, seed: int = 0):
        self.__row = row
        self.__col = col
        self.__seed: int = seed
        self.grid: torch.BoolTensor = torch.ones((row, col), dtype=bool)  # True for a wall and False for a path
        self.list_path: list = []
        self.start_point: list[int, int] = []
        self.end_point: list[int, int] = []
        self.reset(seed)

    def reset(self, seed: Optional[int] = None) -> None:
        self.__seed = (seed, self.__seed)[seed is None]
        self.grid: torch.Tensor = torch.ones((self.__row, self.__col,))  # we first generate a grid full of wall
        row, col = random.randint(1, self.__row-1), random.randint(1, self.__col-1)
        self.start_point = [row, col]
        self.grid[row][col] = False  # we remove the wall from the start point
        self.generation(row, col)  # we recursively generate a maze
        self.end_point = self.list_path[random.randint(0, len(self.list_path)-1)]

    def generation(self, row: int, col: int) -> None:
        random_directions = [1, 2, 3, 4]
        random.shuffle(random_directions)
        for random_direction in random_directions:
            if random_direction == 1:  # up
                if row - 2 <= 0:
                    continue
                if self.grid[row - 1][col] == 1 and self.grid[row - 2][col] == 1:
                    self.grid[row - 1][col] = 0
                    self.grid[row - 2][col] = 0
                    self.list_path.append([row - 1, col])
                    self.list_path.append([row - 2, col])
                    self.generation(row - 2, col)
                continue
            if random_direction == 2:  # down
                if row + 2 >= self.__row - 1:
                    continue
                if self.grid[row + 1][col] == 1 and self.grid[row + 2][col] == 1:
                    self.grid[row + 1][col] = 0
                    self.grid[row + 2][col] = 0
                    self.list_path.append([row + 1, col])
                    self.list_path.append([row + 2, col])
                    self.generation(row + 2, col)
                continue
            if random_direction == 3:  # left
                if col - 2 <= 0:
                    continue
                if self.grid[row][col - 1] == 1 and self.grid[row][col - 2] == 1:
                    self.grid[row][col - 1] = 0
                    self.grid[row][col - 2] = 0
                    self.list_path.append([row, col - 1])
                    self.list_path.append([row, col - 2])
                    self.generation(row, col - 2)
                continue
            if random_direction == 4:  # right
                if col + 2 >= self.__col - 1:
                    continue
                if self.grid[row][col + 1] == 1 and self.grid[row][col + 2] == 1:
                    self.grid[row][col + 1] = 0
                    self.grid[row][col + 2] = 0
                    self.list_path.append([row, col + 2])
                    self.list_path.append([row, col + 2])
                    self.generation(row, col + 2)
        self.end_point = [row, col]

    def render(self, mode: str = "computed") -> None:
        if mode == "computed":
            for i in range(self.__row):
                for j in range(self.__col):
                    if self.start_point == [i, j]:
                        print("S", end="")
                    elif self.end_point == [i, j]:
                        print("E", end="")
                    elif self.grid[i][j] == 1:
                        print("#", end="")
                    elif self.grid[i][j] == 0:
                        print(".", end="")
                print()
        elif mode == "human":
            print("human")

    '''    
    def __reward(self) -> None:
        return None

    def __observation(self) -> None:
        return None

    def step(self) -> None:
        return None
    '''


def main():
    # création du plateau
    board = SimpleMaze(10, 10, 3)

    # Affichage du plateau
    board.render()


if __name__ == '__main__':
    main()

#%%
