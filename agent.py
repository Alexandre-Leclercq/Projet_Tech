import torch
import random
import time
from IPython.display import clear_output
from typing import Optional
from environment import SimpleMaze


def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█', printEnd = "\r"):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end = printEnd)
    # Print New Line on Complete
    if iteration == total:
        print()

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

    def __init__(
            self,
            env: SimpleMaze,
            gamma: int,
            n_min: int,
            q_min: int,
            debug: bool = False,
            polynomial_features_degree: int = 2
    ):
        self.__debug: bool = debug
        self.__env = env
        self.__s = torch.tensor([])
        self.__a = None
        self.__r = None
        self.__polynomial_features_degree = polynomial_features_degree
        self.__n_min = n_min
        self.__q_min = q_min
        self.__trials = 0
        self.__gamma = gamma
        self.__state_index: list = []
        self.__beta = torch.tensor([], dtype=torch.double)
        self.__Nsa = torch.tensor([], dtype=torch.int)

    def __alpha(self, n: int) -> float:
        return (self.__trials / 10) / (self.__trials / 10 + n)

    # x un vecteur de variable de taille 2
    # n le degré du polynome
    def generate_polynomial_normalize_features(self, x: list, biais=True):
        n = self.__polynomial_features_degree
        if len(x) != 2:
            raise Exception("erreur de dimension pour x")
        features = x.copy()
        for i in range(n+1):
            features.append(x[0]**(n-i) * x[1]**i)
        if sum(features) != 0:
            features = [float(i)/sum(features) for i in features]
        if biais:
            features.insert(0, 1)
        return torch.tensor(features, dtype=torch.double)

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
        u = torch.zeros(number_states, dtype=torch.double)
        for i in range(number_states):
            u[i] = torch.max(self.__q_b(i))
        return u

    def __rand_argmax(self, tensor):
        max_inds, = torch.where(tensor == tensor.max())
        random_index = random.randint(0, len(max_inds)-1)
        return max_inds[random_index]

    def __q_b(self, s: torch.Tensor, a=None):
        #if s == [0, 9, 0, 0, 81]:
            #print(s)
            #print(self.__beta)
        if a is None:  # calculate the vector [Q_b[s0], ... Q_b[sn]]
            #print(self.__beta)
            #print(s)
            return torch.matmul(self.__beta, s)
        else:  # calculate Q_b[s, a]
            return torch.matmul(self.__beta[a], s)

    def q_learning_agent(self, s_prime: torch.Tensor, reward_prime: float):
        if s_prime.tolist() not in self.__state_index:  # keep in memory the index associate to the state s_prime
            self.__state_index.append(s_prime.tolist())
            self.__Nsa = torch.cat((self.__Nsa, torch.zeros((1, len(self.__env.actions())))), 0)  # initialise Nsa[s']

        s_prime_index = self.__state_index.index(s_prime.tolist())

        if len(self.__s) != 0:
            s_index = self.__state_index.index(self.__s.tolist())
            self.__Nsa[s_index][self.__a] += 1
            #print("s': "+str(s_prime))
            #print("reward: "+str(reward_prime))
            #print("variation: "+str((self.__r + self.__gamma * torch.max(self.__q_b(s_prime)) - self.__q_b(self.__s, self.__a))))
            #print("")
            #print("max Q_b[s']: "+str(torch.max(self.__q_b(s_prime))))
            #print("Q_b[s][a]"+str(self.__q_b(self.__s, self.__a)))
            self.__beta[self.__a] += self.__alpha(self.__Nsa[s_index][self.__a]) * \
                                     (self.__r + self.__gamma * torch.max(self.__q_b(s_prime)) - self.__q_b(self.__s, self.__a)) \
                                     * self.__s
        #print("Q[s']: "+str(self.function_exploration(self.__q_b(s_prime), self.__Nsa[s_prime_index])))
        #print("\n\n")
        print("Qb[s'] = "+str(self.function_exploration(self.__q_b(s_prime), self.__Nsa[s_prime_index])))
        self.__a = self.__rand_argmax(self.function_exploration(self.__q_b(s_prime), self.__Nsa[s_prime_index]))
        print("Qb[s', a] = "+str(self.__q_b(s_prime, self.__a)))

        self.__s = s_prime

        self.__r = reward_prime

        return self.__env.actions()[self.__a]

    def __reset(self, trials: int, s0):
        self.__trials = trials
        self.__s = torch.tensor([])
        self.__a = None
        self.__r = None
        self.__state_index: list = []
        self.__beta = torch.zeros((len(self.ACTIONS), len(s0)), dtype=torch.double)
        self.__Nsa = torch.tensor([], dtype=torch.int)

    def learning(self, trials: int):
        current_trials: int = 0
        s0 = self.__env.reset()
        s0 = self.generate_polynomial_normalize_features(s0)
        self.__reset(trials, s0)
        action = self.q_learning_agent(s0, self.__env.reward())
        printProgressBar(current_trials, self.__trials)
        nombre_coup = 0
        while current_trials < self.__trials:
            s_prime, reward, done_stage = self.__env.step(action)
            s_prime = self.generate_polynomial_normalize_features(s_prime)
            action = self.q_learning_agent(s_prime, reward)
            #   time.sleep(.1)
            nombre_coup += 1
            if done_stage:
                print("\n\n")
                print("iteration: "+str(current_trials))
                print("nombre_coup: "+str(nombre_coup))
                printProgressBar(current_trials, self.__trials)
                self.__s = torch.tensor([])
                s0 = self.__env.reset()
                s0 = self.generate_polynomial_normalize_features(s0)
                action = self.q_learning_agent(s0, self.__env.reward())
                current_trials += 1
                nombre_coup = 0
        print("learning completed")

    def play(self, mode="computed"):
        s0 = self.__env.reset()
        s0 = self.generate_polynomial_normalize_features(s0)
        action = self.q_learning_agent(s0, self.__env.reward(), False)
        while True:
            s_prime, reward, done_stage = self.__env.step(action)
            s_prime = self.generate_polynomial_normalize_features(s_prime)
            clear_output(wait=False)
            self.__env.render(mode)
            time.sleep(1)

            action = self.q_learning_agent(s_prime, reward, done_stage)

            if done_stage:
                print("Partie terminée")
                break
