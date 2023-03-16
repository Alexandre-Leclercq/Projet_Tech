import torch
import random
import time
from IPython.display import clear_output
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
            gamma: int,
            seed: Optional[int] = 0,
            random_policy: bool = False,
            debug: bool = False
    ):
        self.__debug: bool = debug
        self.__randomPolicy: bool = random_policy
        self.__env = env
        self.__s = None  # actual state
        self.__trials = 0
        self.__gamma = gamma
        self.__tab_utilities: list = []
        self.__Ns: list = []
        self.__tab_visited_state: list = []
        random.seed(seed)

    """
    see p.702 Artificial Intelligence: A modern approach 
    """

    def __alpha(self, n: int) -> float:
        return (self.__trials / 10) / (self.__trials / 10 + n)

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
        if rand < (1 - (current_trial / self.__trials) * 0.75):  # when current_trial --> trials. p --> 0.25
            action = right_policy
        else:
            wrong_action: list = self.__env.actions()
            wrong_action.remove(right_policy)
            wrong_action: str = wrong_action[random.randint(0, len(wrong_action) - 1)]
            action = wrong_action
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

    def __reset(self, trials: int):
        self.__trials = trials
        self.__tab_utilities: list = []
        self.__Ns: list = []
        self.__tab_visited_state: list = []
        self.__s = None

    def learning(self, trials: int):
        self.__reset(trials)
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
        print("learning completed")

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

    def __init__(self, env: SimpleMaze, gamma: int, n_min: int, q_min: int, debug: bool = False):
        self.__debug: bool = debug
        self.__env = env
        self.__s = None  # actual state
        self.__a = None
        self.__r = None
        self.__n_min = n_min
        self.__q_min = q_min
        self.__trials = 0
        self.__gamma = gamma
        self.__Q_table = torch.tensor([], dtype=torch.float)
        self.__state_index: list = []
        self.__Nsa = torch.tensor([], dtype=torch.int)

    def __alpha(self, n: int) -> float:
        alpha = (self.__trials / 10) / (self.__trials / 10 + n)
        return alpha

    def function_exploration(self, q, n: torch.tensor):
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
        if s_prime not in self.__state_index:  # keep in memory the index associate to the state s_prime
            self.__state_index.append(s_prime)
            self.__Q_table = torch.cat((self.__Q_table, torch.zeros((1, len(self.__env.actions())))),
                                       0)  # we add the row for Q[s']
            self.__Nsa = torch.cat((self.__Nsa, torch.zeros((1, len(self.__env.actions())))),
                                   0)  # we add the row for Nsa[s']

        s_prime_index = self.__state_index.index(s_prime)

        if done:  # s_prime is a final state
            self.__Q_table[s_prime_index] = torch.full_like(self.__Q_table[s_prime_index], reward_prime)

        a_prime = torch.argmax(self.__Q_table[s_prime_index])
        if self.__s is not None:
            s_index = self.__state_index.index(self.__s)
            self.__Nsa[s_index][self.__a] = self.__Nsa[s_index][self.__a] + 1
            self.__Q_table[s_index][self.__a] = self.__Q_table[s_index][self.__a] + \
                                                self.__alpha(self.__Nsa[s_index][self.__a]) * \
                                                (self.__r + self.__gamma * self.__Q_table[s_prime_index][a_prime] -
                                                 self.__Q_table[s_index][self.__a])

        self.__s = s_prime
        self.__a = torch.argmax(self.function_exploration(self.__Q_table[s_prime_index], self.__Nsa[s_prime_index]))
        self.__r = reward_prime

        return self.__env.actions()[self.__a]

    def __reset(self, trials: int):
        self.__trials = trials
        self.__s = None  # actual state
        self.__a = None
        self.__r = None
        self.__Q_table = torch.tensor([], dtype=torch.float)
        self.__state_index: list = []
        self.__Nsa = torch.tensor([], dtype=torch.int)

    def learning(self, trials: int):
        current_trials: int = 0
        self.__reset(trials)
        s0 = self.__env.reset()
        action = self.q_learning_agent(s0, self.__env.reward(), False)
        while current_trials < self.__trials:
            s_prime, reward, done_stage = self.__env.step(action)
            action = self.q_learning_agent(s_prime, reward, done_stage)
            if self.__debug:
                self.__debug_env()

            if done_stage:
                self.__s = None
                s0 = self.__env.reset()
                action = self.q_learning_agent(s0, self.__env.reward(), False)
                current_trials += 1
        print("learning completed")

    def play(self, mode="computed"):
        s0 = self.__env.reset()
        action = self.q_learning_agent(s0, self.__env.reward(), False)
        while True:
            s_prime, reward, done_stage = self.__env.step(action)
            clear_output(wait=False)
            self.__env.render(mode)
            time.sleep(1)

            action = self.q_learning_agent(s_prime, reward, done_stage)

            if done_stage:
                print("Partie terminée")
                break


class ActiveAgentRegressionLearning:
    ACTIONS: tuple = (
        "north",
        "east",
        "south",
        "west"
    )

    def __init__(self, env: SimpleMaze, gamma: int, n_min: int, q_min: int, debug: bool = False):
        self.__debug: bool = debug
        self.__env = env
        self.__s = None
        self.__a = None
        self.__r = None
        self.__n_min = n_min
        self.__q_min = q_min
        self.__trials = 0
        self.__gamma = gamma
        self.__state_index: list = []
        self.__beta = torch.tensor([], dtype=torch.float)
        self.__Nsa = torch.tensor([], dtype=torch.int)

    def __alpha(self, n: int) -> float:
        return (self.__trials) / (self.__trials + n)

    def function_exploration(self, q, n: int):
        tmp = torch.zeros(len(q))
        for i in torch.arange(len(q)):
            if n[i] <= self.__n_min:
                tmp[i] = self.__q_min
            else:
                tmp[i] = q[i]
        return tmp

    def get_utilities(self):
        number_states = self.__env.get_number_state()
        u = torch.zeros(number_states, dtype=torch.float)
        for i in range(number_states):
            u[i] = torch.max(self.__q_b(i))
        return u

    def __s_tilde(self, s):
        return torch.hstack((torch.tensor(1, dtype=torch.float), torch.tensor(s, dtype=torch.float)))
    def __q_b(self, s: list, a=None):
        if a is None:  # calculate the vector [Q_b[s0], ... Q_b[sn]]
            return torch.matmul(self.__beta, self.__s_tilde(s))
        else:  # calculate Q_b[s, a]
            return torch.matmul(self.__beta[a], self.__s_tilde(s))

    def q_learning_agent(self, s_prime: list, reward_prime: float, done: bool):
        if s_prime not in self.__state_index:  # keep in memory the index associate to the state s_prime
            self.__state_index.append(s_prime)
            self.__Nsa = torch.cat((self.__Nsa, torch.zeros((1, len(self.__env.actions())))), 0)  # initialise Nsa[s']

        s_prime_index = self.__state_index.index(s_prime)

        if self.__s is not None:
            s_index = self.__state_index.index(self.__s)
            self.__Nsa[s_index][self.__a] += 1
            print("alpha: "+str(self.__alpha(self.__Nsa[s_index][self.__a])))
            print("variation: "+str(self.__alpha(self.__Nsa[s_index][self.__a]) \
                                    * \
                                    (self.__r + self.__gamma * torch.max(self.__q_b(s_prime)) - self.__q_b(self.__s, self.__a)) \
                                    * self.__s_tilde(self.__s)))
            self.__beta[self.__a] += self.__alpha(self.__Nsa[s_index][self.__a]) \
                                     * \
                                     (self.__r + self.__gamma * torch.max(self.__q_b(s_prime)) - self.__q_b(self.__s, self.__a)) \
                                     * self.__s_tilde(self.__s)

        self.__s = s_prime
        self.__a = torch.argmax(self.__q_b(s_prime))
        self.__r = reward_prime

        return self.__env.actions()[self.__a]

    def __reset(self, trials: int, s0):
        self.__trials = trials
        self.__s = None
        self.__a = None
        self.__r = None
        self.__state_index: list = []
        self.__beta = torch.rand((len(self.ACTIONS), len(s0)+1),)*1000
        self.__Nsa = torch.tensor([], dtype=torch.int)

    def learning(self, trials: int):
        current_trials: int = 0
        s0 = self.__env.reset()
        self.__reset(trials, s0)
        print("s0: "+str(s0))
        action = self.q_learning_agent(s0, self.__env.reward(), False)
        while current_trials < self.__trials:
            s_prime, reward, done_stage = self.__env.step(action)
            action = self.q_learning_agent(s_prime, reward, done_stage)

            if done_stage:
                self.__s = None
                s0 = torch.tensor(self.__env.reset(), dtype=torch.float)
                action = self.q_learning_agent(s0, self.__env.reward(), False)
                current_trials += 1
        print("learning completed")

    def play(self, mode="computed"):
        s0 = self.__env.reset()
        action = self.q_learning_agent(s0, self.__env.reward(), False)
        while True:
            s_prime, reward, done_stage = self.__env.step(action)
            clear_output(wait=False)
            self.__env.render(mode)
            time.sleep(1)

            action = self.q_learning_agent(s_prime, reward, done_stage)

            if done_stage:
                print("Partie terminée")
                break

def main():
    print("Hello World!")


if __name__ == '__main__':
    main()

# %%
