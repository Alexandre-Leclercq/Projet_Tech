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
    def __init__(self, env: SimpleMaze, trials: int, gamma: int, debug: bool = False):
        self.__debug: bool = debug
        self.__env = env
        self.__s = None  # actual state
        self.__a = None  # actual state
        self.__r = None  # actual state
        self.__trials = trials
        self.__Q_table: list = torch.zeros((1,len(self.ACTIONS)))
        # self.__R_table: list =[]
        self.__tab_utilities: list = []
        self.__Nsa: list = torch.zeros((....))
        self.__tab_visited_state: list = []


    def __alpha(self, n: int) -> float:
        return self.__trials / (self.__trials + n)

    def function_exploration(self, index_s_prime):
        n = self.__Ns[index_s_prime]
        Nmin = np.argmin(self.__Ns)
        Q = self.__Q_table[index_s_prime]
        Qmin = np.argmin(self.__Q_table)

        if n <= Nmin:
            return Qmin
        else:
            return Q

    def Q_learning_Agent(self, s_prime, reward: float):

        if self.__s is not None:
            index_s = self.__tab_visited_state.index(self.__s)
            index_s_prime = self.__tab_visited_state.index(s_prime)
            self.__Nsa[index_s, self.__a] = self.__Nsa[index_s, self.__a] + 1
#vstack en torch
            self.__Q_table(index_s,self.__a)
            self.__tab_utilities[index_s].append(
                self.__tab_utilities[index_s][-1] + self.__alpha(self.__Nsa[index_s, self.__a]) *
                (reward + self.__gamma * self.__tab_utilities[index_s_prime][-1]
                 - self.__tab_utilities[index_s][-1]))

        self.__s = s_prime
        # self.__R_table[index_s] = reward
        self.__r = reward
        return self.function_exploration(index_s_prime)
