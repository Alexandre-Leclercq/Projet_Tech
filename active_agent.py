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
    def __init__(self, env: SimpleMaze, trials: int, gamma: int, n_min: int, q_min:int, debug: bool = False):
        self.__debug: bool = debug
        self.__env = env
        self.__s = None  # actual state
        self.__a = None
        self.__r = None
        self.__n_min = n_min
        self.__q_min = q_min
        self.__trials = trials
        self.__gamma = gamma
        self.__Q_table = torch.zeros((self.__env.get_number_state(), len(self.ACTIONS)))
        self.__tab_utilities: list = []
        self.__Nsa = torch.zeros((1, len(self.ACTIONS)))
        self.__tab_visited_state: list = []

    def __alpha(self, n: int) -> float:
        return self.__trials / (self.__trials + n)

    def function_exploration(self, q, n):
        tmp = torch.zeros(size=(1, len(q)))
        for i in range(len(q)):

            if n <= self.__n_min[i]:
                tmp[i] = self.__q_min[i]
            else:
                tmp[i] = q[i]

        return tmp

    def Q_learning_Agent(self, s_prime, reward_prime: float):

        a_prime = torch.argmax(self.__Q_table[s_prime])

        if self.__s is not None:
            index_s = self.__tab_visited_state.index(self.__s)
            self.__Nsa[index_s][self.__a] = self.__Nsa[index_s][self.__a] + 1
            self.__Q_table[index_s][self.__a] = self.__Q_table[index_s][self.__a] + self.__alpha(self.__Nsa[index_s][self.__a])
            (self.__r + self.__gamma * torch.argmax(self.__Q_table[s_prime][a_prime]-self.__Q_table[self.__s][self.__a]))

        self.__s = s_prime
        self.__a = torch.argmax(self.function_exploration(self.__Q_table[s_prime], self.__Nsa[s_prime]))
        self.__r = reward_prime

        return self.ACTIONS[self.__a]

    def learning(self):
        current_trials: int = 0
        s0 = self.__env.reset()
        action = self.Q_learning_Agent(s0, self.__env.reward())
        while current_trials < self.__trials:
            s_prime, reward, done_stage = self.__env.step(action)
            action = self.Q_learning_Agent(s_prime, reward)

            if done_stage:
                self.__s = None
                s0 = self.__env.reset()
                action = self.Q_learning_Agent(s0, self.__env.reward())
                current_trials += 1
