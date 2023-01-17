# -*- coding: utf-8 -*-
"""
Created on Wed Dec 21 10:45:30 2022

@author: adcol
"""


class Board:
    def __init__(self, columns: int, rows: int, tabExit: list):
        self.__columns = columns
        self.__rows = rows
        self.__character = None
        self.__grille = [[0 for j in range(columns)] for i in range(rows)]
        self.__obstacles = []
        self.__sortie = tabExit
        self.__trou = []
        self.__etoile=[]

        
    def display(self) -> None:
        for y in range(self.__rows):
            for x in range(self.__columns):
                if self.__character is not None and self.__character.position == [x, y]:
                    print("C", end=" ")
                elif [x, y] in self.__obstacles:
                    print("O", end=" ")
            
                elif [x, y] in self.__trou:
                    print("T", end=" ")
        
                elif [x, y] in self.__etoile:
                    print("*", end=" ")
        
                elif self.__sortie == [x, y]:
                    print("S", end=" ")
                else:
                    print("-", end=" ")
                print()

    def calculposManhattan(self):
        sortiecol = self.__sortie[0]
        sortierow = self.__sortie[1]
        charcol = self.__character.position[0]
        charrow = self.__character.position[1]
        Score = abs(sortiecol-charcol) + abs(sortierow-charrow)
        return Score
         
    def get_largeur(self):
        return self.__largeur
   
    def get_hauteur(self):
        return self.__hauteur
   
    def get_cellule(self, x, y):
        return self.__grille[y][x]
   
    def debut_jeu(self, board):
        jeu: bool = True
        nbCoups: int = 0
        test: list = [0, 0]
        while jeu:
            if self.character.position == self.sortie:
                print("fin jeu")
                board.display()
                score = board.calculposManhattan()
                nbCoups += test[0]
                Total = score + nbCoups
                print("\n Heuristiqu : ", score, "\n Nb coups joué + malus", nbCoups,
                      "\n Malus du coups :", test[0], "\n Bonus du coups :", test[1],
                      "\n cout Total : ", Total)
                nbCoups += 1
                jeu = True
                break
           
            board.display()
            score = board.calculposManhattan()
           
            Total = score + nbCoups + test[0]-test[1]
            print("\n Score : ", score, "\n Nb coups joué", nbCoups,
                  "\n Malus du coups :", test[0], "\n Bonus du coups :", test[1],
                  "\n cout Total : ", Total)
            nbCoups +=1
            test = board.character.ask_move(board)
            print("\n==================== \n")

           

class Character:

    def __init__(self, position):
        self.__position = position


    def move(self, dx, dy, board):
        new_position = [self.__position[0] + dx, self.__position[1] + dy]
        if new_position not in board.obstacles:
            self.__position = new_position

    def ask_move(self, board):
        malus=0
        bonus=0
        while True:
            move = input("Enter move (up, down, left, right): ")
            if move == "up":
                dx, dy = 0, -1
            elif move == "down":
                dx, dy = 0, 1
            elif move == "left":
                dx, dy = -1, 0
            elif move == "right":
                dx, dy = 1, 0
            else:
                print("Invalid move")
                continue
        
        
            new_position = [self.position[0] + dx, self.position[1] + dy]
            if new_position in board.obstacles:
                print("Invalid move")
                continue
          
            if new_position in board.trou:
                print("Tomber dans un trou malus 2 mvt")
                malus+=2
              
            if new_position in board.etoile:
                print("Trouve étoile bonus -2 mvt")
                bonus +=2
              
    
            if self.__position[0] + dx < 0 or self.__position[0] + dx >= board.columns:
                print("Invalid move")
                continue
            if self.__position[1] + dy < 0 or self.__position[1] + dy >= board.rows:
                print("Invalid move")
                continue
    
            self.move(dx, dy, board)
            test = [malus, bonus]
            return test
            break


def main():
    # création du plateau
    board = Board(10, 10, [6, 2])

    player = Character([0, 2]) # colonne rows
    board.character = player

    # ajout des obstacles colonnes lignes
    board.obstacles = [[1, 1], [2, 1], [3, 1]]
    board.trou = [[5, 2], [4, 4]]
    board.etoile = [[2, 2], [6, 4]]

    # Affichage du plateau
    board.debut_jeu(board)


if __name__ == '__main__':
    main()



