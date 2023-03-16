import random
import time
from ipywidgets import Image
from ipycanvas import Canvas

character_sprite = Image.from_file("Assets/amongus_character.png")
chest_sprite = Image.from_file("Assets/chest.png")
open_chest_sprite = Image.from_file("Assets/open_chest.png")
path_sprite = Image.from_file("Assets/path6.png")
wall_sprite = Image.from_file("Assets/wall.png")
"""
paths_sprite = [Image.from_file("Assets/path1.png"),
                Image.from_file("Assets/path2.png"),
                Image.from_file("Assets/path3.png"),
                Image.from_file("Assets/path4.png"),
                Image.from_file("Assets/path5.png"),
                Image.from_file("Assets/path6.png"),
                Image.from_file("Assets/path7.png"),
                Image.from_file("Assets/path8.png"),
                Image.from_file("Assets/path9.png"),
                Image.from_file("Assets/path10.png"),
                Image.from_file("Assets/path11.png"),
                Image.from_file("Assets/path12.png")]
"""
spikes_sprite = Image.from_file("Assets/spikes.png")
coin_sprite = Image.from_file("Assets/coin.png")


class CanvasInterface:

    def __init__(self, grid: list, cell_types: dict, cell_size: int, end_pos: list, character_pos: list):
        self.__grid: list = grid
        self.__row: int = len(grid)
        self.__col: int = len(grid[0])
        self.__cs: int = cell_size
        self.__cell_types: dict = cell_types
        self.__character_pos: list = character_pos
        self.__end_pos: list = end_pos
        self.__canvas: Canvas = Canvas(width=self.__col * self.__cs, height=self.__row * self.__cs)

    def initialize_canvas(self):
        for i in range(self.__row):
            for j in range(self.__col):
                self.draw_cell(i, j)
        self.draw_end_cell()
        self.__canvas.draw_image(character_sprite, self.__character_pos[1] * self.__cs,
                                 self.__character_pos[0] * self.__cs, self.__cs, self.__cs)
        self.print_canvas()

    def print_canvas(self):
        display(self.__canvas)

    def clear_character(self):
        self.__canvas.clear_rect(self.__character_pos[1] * self.__cs, self.__character_pos[0] * self.__cs, self.__cs,
                                 self.__cs)
        self.draw_cell(self.__character_pos[0], self.__character_pos[1])

    def clear_canvas(self):
        self.__canvas.clear()

    def draw_cell(self, i: int, j: int):
        if self.__grid[i][j] == self.__cell_types["spikes"]:
            self.__canvas.draw_image(spikes_sprite, j * self.__cs, i * self.__cs, self.__cs, self.__cs)
        elif self.__grid[i][j] == self.__cell_types["coin"]:
            self.__canvas.draw_image(path_sprite, j * self.__cs, i * self.__cs, self.__cs,
                                     self.__cs)
            self.__canvas.draw_image(coin_sprite, j * self.__cs, i * self.__cs, self.__cs, self.__cs)
        elif self.__grid[i][j] == self.__cell_types["wall"]:
            self.__canvas.draw_image(wall_sprite, j * self.__cs, i * self.__cs, self.__cs,
                                     self.__cs)
        else:
            self.__canvas.draw_image(path_sprite, j * self.__cs, i * self.__cs, self.__cs,
                                     self.__cs)
        self.__canvas.stroke_rect(j * self.__cs, i * self.__cs, self.__cs, self.__cs)

    def draw_canvas(self, new_pos: list, grid: list):
        self.__grid = grid
        self.clear_canvas()
        for i in range(self.__row):
            for j in range(self.__col):
                self.draw_cell(i, j)
        self.draw_end_cell()
        self.animation(new_pos[0], new_pos[1])
        self.draw_character(new_pos)

    def draw_character(self, new_pos: list):
        self.__canvas.draw_image(character_sprite, new_pos[1] * self.__cs, new_pos[0] * self.__cs, self.__cs, self.__cs)
        self.__canvas.stroke_rect(new_pos[1] * self.__cs, new_pos[0] * self.__cs, self.__cs, self.__cs)
        self.__character_pos = [new_pos[0], new_pos[1]]

    def animation(self, x: int, y: int):
        if self.__grid[x][y] == 3:
            self.__canvas.draw_image(open_chest_sprite, y * self.__cs, x * self.__cs, self.__cs, self.__cs)
        elif self.__grid[x][y] == 2:
            self.__canvas.draw_image(path_sprite, y * self.__cs, x * self.__cs, self.__cs, self.__cs)

    def draw_end_cell(self):
        self.__canvas.draw_image(path_sprite, self.__end_pos[1] * self.__cs, self.__end_pos[0] * self.__cs, self.__cs,
                                 self.__cs)
        self.__canvas.draw_image(chest_sprite, self.__end_pos[1] * self.__cs, self.__end_pos[0] * self.__cs, self.__cs, self.__cs)
        self.__canvas.stroke_rect(self.__end_pos[1] * self.__cs, self.__end_pos[0] * self.__cs, self.__cs, self.__cs)


def main():
    print("Hello World!")


if __name__ == '__main__':
    main()
