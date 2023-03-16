#%%
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 21 10:45:30 2022

@author: Sébastien CIVADE, Adrien COLMART, Thomas FOY, Alexandre LECLERCQ
"""
import random
import torch
from canvasInterface import CanvasInterface
from abc import ABC, abstractmethod
from typing import Optional


class Environment(ABC):

    """
    reset the environment
    """
    @abstractmethod
    def reset(self):
        pass

    """
    execute one transition in the environment with the action given
    """
    @abstractmethod
    def step(self, action):
        pass

    """
    return the current state
    """
    @abstractmethod
    def state(self):
        pass

    """
    return the reward for the current state of the environment
    """
    @abstractmethod
    def reward(self):
        pass

    @abstractmethod
    def actions(self):
        pass


class SimpleMaze(Environment):

    ACTIONS: dict = {  # we define the different actions doable
        "north": (-1, 0),
        "east": (0, 1),
        "south": (1, 0),
        "west": (0, -1)
    }

    def __init__(self, row: int, col: int, seed: int = 0):
        self.__row = row
        self.__col = col
        self.__seed: int = seed
        self.character_pos: list = [row - 1, 0]
        self.end_point: tuple = [0, col - 1]
        #  self.reset(seed)

    def actions(self) -> list:
        return list(self.ACTIONS.keys())

    def reset(self, seed: Optional[int] = None) -> list:
        self.__seed = (seed, self.__seed)[seed is None]
        random.seed(self.__seed)
        self.character_pos: list = [self.__row - 1, 0]
        return self.state()


    def done(self) -> bool:
        return self.character_pos == self.end_point

    def reward(self) -> float:
        if self.character_pos == self.end_point:
            return 1000
        else:
            return -1



    """
    return the state as a unique integer
    """
    def state(self) -> list:
        return self.character_pos.copy()

    def get_number_state(self):
        return self.__row * self.__col

    def step(self, action: int) -> (list, int, bool):
        movement = self.ACTIONS[action]
        if self.__row > self.character_pos[0] + movement[0] >= 0 and self.__col > self.character_pos[1] + movement[1] >= 0:
            print("ok1")
            self.character_pos[0] += movement[0]
            self.character_pos[1] += movement[1]
        return self.state(), self.reward(), self.done()

    def render(self, mode: str = "computed") -> None:
        if mode == "computed":
            for i in torch.arange(self.__row):
                print("{:<2}".format(str(i.item())), end=" ")
                for j in torch.arange(self.__col):
                    print("|", end="")
                    if self.character_pos == [i, j]:
                        print("C", end="")
                    elif self.end_point == [i, j]:
                        print("E", end="")
                    else:
                        print(".", end="")
                print("|")
            print("")
        elif mode == "human":  # futur gui render mode
            print("human")


class OldMaze(Environment):  # Maze environment base on the depth first search algorithm

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
        self.paths: list = []
        self.character_pos: list = []
        self.end_point: list = []
        self.reset(seed)

    def reset(self, seed: Optional[int] = None) -> None:
        self.__seed = (seed, self.__seed)[seed is None]
        row, col = random.randint(1, self.__row-1), random.randint(1, self.__col-1)
        self.character_pos = [row, col]  # we set the character at a random starting point
        self.paths = [self.character_pos[:]]  # we set the starting point as an empty cell
        self.generation(row, col)  # we recursively generate a maze
        self.end_point = self.paths[random.randint(0, len(self.paths)-1)].copy()  # we take a random empty cell set it as the end point

    def actions(self) -> list:
        return list(self.ACTIONS.keys())

    def generation(self, row: int, col: int) -> None:
        random_directions = [1, 2, 3, 4]
        random.shuffle(random_directions)
        for random_direction in random_directions:
            if random_direction == 1:  # up
                if row - 2 <= 0:
                    continue
                if [row - 1, col] not in self.paths and [row - 2, col] not in self.paths:
                    self.paths.append([row - 1, col])
                    self.paths.append([row - 2, col])
                    self.generation(row - 2, col)
                continue
            if random_direction == 2:  # down
                if row + 2 >= self.__row - 1:
                    continue
                if [row + 1, col] not in self.paths and [row + 2, col] not in self.paths:
                    self.paths.append([row + 1, col])
                    self.paths.append([row + 2, col])
                    self.generation(row + 2, col)
                continue
            if random_direction == 3:  # left
                if col - 2 <= 0:
                    continue
                if [row, col - 1] not in self.paths and [row, col - 2] not in self.paths:
                    self.paths.append([row, col - 1])
                    self.paths.append([row, col - 2])
                    self.generation(row, col - 2)
                continue
            if random_direction == 4:  # right
                if col + 2 >= self.__col - 1:
                    continue
                if [row, col + 1] not in self.paths and [row, col + 2] not in self.paths:
                    self.paths.append([row, col + 1])
                    self.paths.append([row, col + 2])
                    self.generation(row, col + 2)

    def get_number_state(self):
        return self.__row * self.__col

    def render(self, mode: str = "computed") -> None:
        if mode == "computed":
            for i in torch.arange(self.__row):
                for j in torch.arange(self.__col):
                    if self.character_pos == [i, j]:
                        print("S", end="")
                    elif self.end_point == [i, j]:
                        print("E", end="")
                    elif [i, j] in self.paths:
                        print(".", end="")
                    else:
                        print("#", end="")

                print()
        elif mode == "human":
            print("human")

    def step(self, action: int) -> (list, int, bool):
        movement = self.ACTIONS[action]
        if self.__row > self.character_pos[0] + movement[0] >= 0 and self.__col > self.character_pos[1] + movement[1] >= 0:
            self.character_pos[0] += movement[0]
            self.character_pos[1] += movement[1]
        return self.state(), self.reward(), self.done()

    def state(self) -> int:
        return self.character_pos.copy()

    def reward(self) -> float:
        if self.character_pos == self.end_point:
            return 1000
        else:
            return -1



def main():
    # création du plateau
    board = SimpleMaze(10, 10, 3)

    # Affichage du plateau
    board.render()


if __name__ == '__main__':
    main()

#%%
class Maze(Environment):

    ACTIONS: dict = {  # we define the different actions doable
        "north": (-1, 0),
        "east": (0, 1),
        "south": (1, 0),
        "west": (0, -1)
    }

    CELLS_TYPE: dict = {
        "empty": 0,
        "spikes": 1,
        "coin":2,
        "wall": 3,
    }

    OBSTACLES_PROPORTION: dict = {
        "spikes": 8,
        "coin": 2
    }



    def __init__(self, row: int, col: int, seed: int = 0, ratio_obstacles: int = 0):
        self.__row = row
        self.__col = col
        self.__seed: int = seed
        self.character_pos: list = [row - 1, 0]
        self.end_point: tuple = []
        self.ratio_obstacles = ratio_obstacles
        self.grid = torch.tensor([])
        self.canvasInterface = CanvasInterface()
        self.reset(seed)

    def actions(self) -> list:
        return list(self.ACTIONS.keys())


    def reset(self, seed: Optional[int] = None) -> list:
        self.__seed = (seed, self.__seed)[seed is None]
        random.seed(self.__seed)
        self.grid = torch.ones((self.__row,self.__col), dtype=torch.int) * self.CELLS_TYPE['wall']
        row, col = random.randint(1, self.__row-1), random.randint(1, self.__col-1)
        #row, col = self.__row - 1, 0
        self.character_pos: list = [row, col]
        self.generation_wall(row, col)
        self.generate_obstacle()
        return self.character_pos.copy()

    def generation_wall(self, row: int, col: int) -> None:
        random_directions = [1, 2, 3, 4]
        random.shuffle(random_directions)
        for random_direction in random_directions:
            if random_direction == 1:  # up
                if row - 2 <= 0:
                    continue
                if self.grid[row - 1][col] != self.CELLS_TYPE['empty'] and self.grid[row - 2][col] != self.CELLS_TYPE['empty']:
                    self.grid[row - 1][col] = self.CELLS_TYPE['empty']
                    self.grid[row - 2][col] = self.CELLS_TYPE['empty']
                    self.generation_wall(row - 2, col)
                continue
            if random_direction == 2:  # down
                if row + 2 >= self.__row - 1:
                    continue
                if self.grid[row + 1][col] != self.CELLS_TYPE['empty'] and self.grid[row + 2][col] != self.CELLS_TYPE['empty']:
                    self.grid[row + 1][col] = self.CELLS_TYPE['empty']
                    self.grid[row + 2][col] = self.CELLS_TYPE['empty']
                    self.generation_wall(row + 2, col)
                continue
            if random_direction == 3:  # left
                if col - 2 <= 0:
                    continue
                if self.grid[row][col - 1] != self.CELLS_TYPE['empty'] and self.grid[row][col - 2] != self.CELLS_TYPE['empty']:
                    self.grid[row][col - 1] = self.CELLS_TYPE['empty']
                    self.grid[row][col - 2] = self.CELLS_TYPE['empty']
                    self.generation_wall(row, col - 2)
                continue
            if random_direction == 4:  # right
                if col + 2 >= self.__col - 1:
                    continue
                if self.grid[row][col + 1] != self.CELLS_TYPE['empty'] and self.grid[row][col + 2] != self.CELLS_TYPE['empty']:
                    self.grid[row][col + 1] = self.CELLS_TYPE['empty']
                    self.grid[row][col + 2] = self.CELLS_TYPE['empty']

                    self.generation_wall(row, col + 2)
        self.end_point = [row, col]

    def generate_obstacle(self):
        free_place = []

        for i in torch.arange(self.__row):
            for j in torch.arange(self.__col):
                if [i, j] == self.character_pos:
                    continue
                if self.grid[i][j].item() == 0:
                    free_place.append([i.item(), j.item()])

        place_obstacle = []
        type_obstacle = []
        random.shuffle(free_place)  # ressort la liste mélanger

        for i in range(int((len(free_place))*self.ratio_obstacles)-1):  # on garde une place de libre pour le endpoint
            random_value = random.randint(1, self.__col-1)
            if random_value <= self.OBSTACLES_PROPORTION['spikes']:
                row, col = free_place.pop()
                self.grid[row][col] = self.CELLS_TYPE['spikes']

            elif random_value <= self.OBSTACLES_PROPORTION['spikes'] + self.OBSTACLES_PROPORTION['coin']:
                row, col = free_place.pop()
                self.grid[row][col] = self.CELLS_TYPE['coin']

        # on ajoute le endpoint
        self.end_point = free_place.pop()

        """print('la boucle qui génère type obs',(int(len(free_place)/8)-1))
        print(len(type_obstacle))
        print('la boucle qui génère place obs',int(len(free_place)/8))
        print(len(place_obstacle))"""


    def done(self) -> bool:
        return self.character_pos == self.end_point

    def reward(self) -> float:
        if self.character_pos == self.end_point:
            return 1000

        if self.grid[self.character_pos[0], self.character_pos[1]] == 1:
            return -200

        if self.grid[self.character_pos[0], self.character_pos[1]] == 2:
            self.grid[self.character_pos[0], self.character_pos[1]] = 0
            return 200
        else:
            return -1



    """
    return the state as a unique integer
    """
    def state(self) -> int:
        return self.character_pos.copy()

    def get_number_state(self):
        return self.__row * self.__col

    def step(self, action: int) -> (list, int, bool):
        movement = self.ACTIONS[action]
        if self.__row > self.character_pos[0] + movement[0] >= 0 and \
                self.__col > self.character_pos[1] + movement[1] >= 0 and\
                self.grid[self.character_pos[0]+movement[0], self.character_pos[1]+movement[1]] != self.CELLS_TYPE['wall']:
            self.character_pos[0] += movement[0]
            self.character_pos[1] += movement[1]
        return self.state(), self.reward(), self.done()

    def render(self, mode: str = "computed") -> None:
        if mode == "computed":
            for i in torch.arange(self.__row):
                print("{:<4}".format(str(i.item())), end=" ")
                for j in torch.arange(self.__col):
                    print("|", end="")
                    if self.character_pos == [i, j]:
                        print("A", end="")
                    elif self.end_point == [i, j]:
                        print("E", end="")
                    elif self.grid[i][j] == self.CELLS_TYPE["spikes"]:
                        print("P", end="")
                    elif self.grid[i][j] == self.CELLS_TYPE["wall"]:
                        print("#", end="")
                    elif self.grid[i][j] == self.CELLS_TYPE["coin"]:
                        print("C", end="")
                    elif self.grid[i][j] == self.CELLS_TYPE["empty"]:
                        print(".", end="")
                print("|")
            print("")
        elif mode == "gui":  # futur gui render mode
            self.canvasInterface.draw(self.grid, self.CELLS_TYPE, cell_size=96, end_pos=self.end_point, character_pos=self.character_pos)



#%%
