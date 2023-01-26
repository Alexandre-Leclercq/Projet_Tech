import torch
import random
from typing import Optional
from environment import SimpleMaze


def actions():
    return ["north", "east", "south", "west"]


class PassiveAgentTD:

    ACTIONS: tuple = (
        "north",
        "east",
        "south",
        "west"
    )

    def __init__(
            self, env: SimpleMaze,
            trials: int,
            gamma: int,
            seed: Optional[int] = 0,
            random_policy: bool = False,
            debug: bool = False
            ):
        self.__debug: bool = debug
        self.__randomPolicy: bool = random_policy
        self.__env = env
        self.__s = None  # actual state
        self.__trials = trials
        self.__gamma = gamma
        self.__tab_utilities: list = []
        self.__Ns: list = []
        self.__tab_visited_state: list = []
        random.seed(seed)

    """
    see p.702 Artificial Intelligence: A modern approach 
    """
    def __alpha(self, n: int) -> float:
        return self.__trials / (self.__trials + n)

    def __update_utility(self, s_prime, reward: float) -> None:  # U[s] + alpha(Ns[s]) (R[s] + γU[s′] − U[s])
        if s_prime not in self.__tab_visited_state:
            self.__tab_utilities.append([0])
            self.__Ns.append(0)
            self.__tab_visited_state.append(s_prime)

        if self.__s is not None:
            index_s = self.__tab_visited_state.index(self.__s)
            index_s_prime = self.__tab_visited_state.index(s_prime)
            self.__Ns[index_s] = self.__Ns[index_s] + 1
            self.__tab_utilities[index_s].append(self.__tab_utilities[index_s][-1] + self.__alpha(self.__Ns[index_s]) *
                                                 (reward + self.__gamma * self.__tab_utilities[index_s_prime][-1]
                                                  - self.__tab_utilities[index_s][-1]))
        if self.__debug:
            self.__debug_env(s_prime, reward)
        self.__s = s_prime



    def __policy(self) -> str:
        if self.__s[0] == 0:  # if we have reach the first row (the top)
            return "east"  # right
        else:
            return "north"  # up

    def __random_policy(self, current_trial: int):
        rand = random.uniform(0, 1)
        right_policy = self.__policy()
        print("right policy: " + right_policy)
        print("random: "+str(rand))
        print("p = "+str(1 - (current_trial/self.__trials) * 0.75))
        if rand < (1 - (current_trial/self.__trials) * 0.75):  # when current_trial --> trials. p --> 0.25
            return right_policy
        else:
            wrong_action: list = list(self.ACTIONS)
            wrong_action.remove(right_policy)
            wrong_action: str = wrong_action[random.randint(0, len(wrong_action)-1)]
            return wrong_action

    def __debug_env(self, s_prime=None, reward=None):
        self.__env.render()
        print("s: " + str(self.__s))
        print("s_prime: " + str(s_prime))
        print("reward: " + str(reward))
        print("U[s] = " + str(self.get_utilities()))
        print("state[s] = " + str(self.get_visited_state()))
        print("N[s] = " + str(self.get_ns()))
        print("\n")

    def learning(self):
        current_trial: int = 0
        s0 = self.__env.reset()
        self.__update_utility(s0, 0)  # we add the utility of the s0 state

        while current_trial < self.__trials:
            if self.__randomPolicy:
                action = self.__random_policy(current_trial)
            else:
                action = self.__policy()
            print("action choisi: "+str(action)+"\n")
            s_prime, reward, done_stage = self.__env.step(action)
            self.__update_utility(s_prime, reward)

            if done_stage:
                self.__s = self.__env.reset()
                if self.__debug:
                    self.__debug_env()
                current_trial += 1

    def print_u_table(self):
        for state in self.get_visited_state():
            print("{:<10}".format(str(state)), end="")
        print()
        for i in range(self.__trials):  # for each trials
            for j in range(len(self.get_utilities())):  # for state of a trials
                if i < len(self.get_utilities()[j]):
                    print("{:<10}".format(str(round(self.__tab_utilities[j][i], 2))), end="")
                    continue
                print("{:<10}".format("None"), end="")
            print()

    def get_utilities(self):
        return self.__tab_utilities[:-1]

    def get_visited_state(self):
        return self.__tab_visited_state[:-1]

    def get_ns(self):
        return self.__Ns[:-1]


def main():
    print("Hello World!")


if __name__ == '__main__':
    main()
