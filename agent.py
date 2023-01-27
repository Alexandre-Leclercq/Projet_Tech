import torch
import random
from typing import Optional
from environment import SimpleMaze


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
        alpha = (self.__trials/10) / (self.__trials/10 + n)
        print("alpha : " + str(alpha))
        return alpha

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
            action = right_policy
        else:
            wrong_action: list = list(self.ACTIONS)
            wrong_action.remove(right_policy)
            wrong_action: str = wrong_action[random.randint(0, len(wrong_action)-1)]
            action = wrong_action
        print("selected action: "+str(action)+"\n")
        return action

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


class ActiveAgentQLearning:

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
        self.__Q_table = torch.zeros((self.__env.get_number_state(), len(self.ACTIONS)), dtype=torch.float)
        self.__tab_utilities: list = []
        self.__Nsa = torch.zeros((self.__env.get_number_state(), len(self.ACTIONS)), dtype=torch.int)
        self.__tab_visited_state: list = []

    def __alpha(self, n: int) -> float:
        alpha = (self.__trials/10) / (self.__trials/10 + n)
        return alpha

    def function_exploration(self, q, n: int):
        tmp = torch.zeros(len(q))
        for i in range(len(q)):
            if n[i] <= self.__n_min:
                tmp[i] = self.__q_min
            else:
                tmp[i] = q[i]
        return tmp

    def __debug_env(self):
        self.__env.render()

    def get_utilities(self):
        number_states = self.__env.get_number_state()
        u = torch.zeros(number_states, dtype=torch.float)
        for i in range(number_states):
            u[i] = torch.max(self.__Q_table[i])
        return u

    def q_learning_agent(self, s_prime, reward_prime: float, done: bool):
        if done:  # s_prime is a final state
            print("r_f: "+str(reward_prime))
            self.__Q_table[s_prime] = torch.full_like(self.__Q_table[s_prime], reward_prime)

        a_prime = torch.argmax(self.__Q_table[s_prime])
        if self.__s is not None:
            print("s_prime: "+str(s_prime))
            print("r: "+str(reward_prime))
            self.__Nsa[self.__s][self.__a] = self.__Nsa[self.__s][self.__a] + 1
            self.__Q_table[self.__s][self.__a] = self.__Q_table[self.__s][self.__a] + \
                                                 self.__alpha(self.__Nsa[self.__s][self.__a]) * \
                                                 (self.__r + self.__gamma *
                                                  (self.__Q_table[s_prime][a_prime]-self.__Q_table[self.__s][self.__a]))

        self.__s = s_prime
        self.__a = torch.argmax(self.function_exploration(self.__Q_table[s_prime], self.__Nsa[s_prime]))
        self.__r = reward_prime

        return self.ACTIONS[self.__a]

    def learning(self):
        current_trials: int = 0
        s0 = self.__env.reset()
        s0 = self.__env.state()
        action = self.q_learning_agent(s0, self.__env.reward(), False)
        while current_trials < self.__trials:
            s_prime, reward, done_stage = self.__env.step(action)
            s_prime = self.__env.state()
            action = self.q_learning_agent(s_prime, reward, done_stage)
            if self.__debug:
                self.__debug_env()

            if done_stage:
                print(current_trials)
                self.__s = None
                s0 = self.__env.reset()
                s0 = self.__env.state()
                action = self.q_learning_agent(s0, self.__env.reward(), False)
                current_trials += 1


def main():
    print("Hello World!")


if __name__ == '__main__':
    main()

#%%
