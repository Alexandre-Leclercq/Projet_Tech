import torch

from environment import SimpleMaze
import numpy as np


class ActiveAgentTD:

    ACTIONS: tuple = (
        "north",
        "east",
        "south",
        "west"
    )
    def __init__(self, env: SimpleMaze, trials: int, gamma: int, Nmin: int, debug: bool = False):
        self.__debug: bool = debug
        self.__env = env
        self.__s = None  # actual state
        self.__a = None  # actual state
        self.__r = None  # actual state
        self.__Nmin= Nmin
        self.__trials = trials
        self.__gamma = gamma
        self.__Q_table = torch.zeros((1,len(self.ACTIONS)))
        # self.__R_table: list =[]
        self.__tab_utilities: list = []
        self.__Nsa = torch.zeros((1,len(self.ACTIONS)))
        self.__tab_visited_state: list = []


    def __alpha(self, n: int) -> float:
        return self.__trials / (self.__trials + n)

    def function_exploration(self,q , n):

        a = self.__a
        #n = torch.sum(self.__Q_table[:][a],0) # ,0 for sum col
        Qmin = #valeur a mettre

        if n <= self.__Nmin:
            return Qmin
        else:
            return q

    def Q_learning_Agent(self, s_prime, reward_prime: float): # voir comment on retrouve l'action effectuer dans la qtable

        if self.__s is not None:
            index_s = self.__tab_visited_state.index(self.__s)
            index_s_prime = self.__tab_visited_state.index(s_prime)
            self.__Nsa[index_s][self.__a] = self.__Nsa[index_s][self.__a] + 1
            self.__Q_table[index_s][self.__a] =  self.__Q_table[index_s][self.__a] + self.__alpha(self.__Nsa[index_s][self.__a])
            (self.__r + self.__gamma * max(self.__Q_table[s_prime][:])-self.__Q_table[self.__s][self.__a])


        if self.__s not in self.__tab_visited_state:
            ajout = torch.zeros((1,len(self.ACTIONS)))
            passe valeur dans ajout
            vstack entre ancien tab et ajout

        self.__s = s_prime
        # self.__R_table[index_s] = reward
        self.__a = max(self.function_exploration(self.__Q_table[s_prime][:],self.__Nsa[s_prime][:]))
        self.__r = reward_prime

        return self.__a
