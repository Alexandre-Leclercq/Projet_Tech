from environment import SimpleMaze


class PassiveAgentTD:
    def __init__(self, env: SimpleMaze, trials: int, gamma: int, debug: bool = False):
        self.__debug: bool = debug
        self.__env = env
        self.__s = None  # actual state
        self.__trials = trials
        self.__gamma = gamma
        self.__tab_utilities: list = []
        self.__Ns: list = []
        self.__tab_visited_state: list = []

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
        if self.__s[0] == 0:
            return "east"  # right
        else:
            return "north"  # up

    """
    def __up_and_right_policy(self):
        if self.__istop:
            return 2  # right
        else:
            return 1  # up
    """

    def __debug_env(self, s_prime=None, reward=None):
        self.__env.render()
        print("s: " + str(self.__s))
        print("s_prime: " + str(s_prime))
        print("reward: " + str(reward))
        print("U[s] = " + str(self.get_utilities()))
        print("state[s] = " + str(self.get_visited_state()))
        print("N[s] = " + str(self.get_Ns()))
        print("\n")

    def learning(self):
        current_trials: int = 0
        s0 = self.__env.reset()
        self.__update_utility(s0, 0)

        while current_trials < self.__trials:
            action = self.__policy()
            s_prime, reward, done_stage = self.__env.step(action)
            self.__update_utility(s_prime, reward)

            if done_stage:
                self.__s = self.__env.reset()
                if self.__debug:
                    self.__debug_env()
                current_trials += 1

    def print_u_table(self):
        for state in self.get_visited_state():
            print("{:<10}".format(str(state)), end="")
        print()
        for i in range(self.__trials):  # for each trials
            for j in range(len(self.get_utilities())):  # for state of a trials
                print("{:<10}".format(str(round(self.__tab_utilities[j][i], 2))), end="")
            print()

    def get_utilities(self):
        return self.__tab_utilities[:-1]

    def get_visited_state(self):
        return self.__tab_visited_state[:-1]

    def get_Ns(self):
        return self.__Ns[:-1]


def main():
    print("Hello World!")


if __name__ == '__main__':
    main()
