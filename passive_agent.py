from environment import SimpleMaze


class Agent:
    def __init__(self, gamma: int, max_iter: int, debug: bool = False):
        self.__debug: bool = debug
        self.__s = None  # actual state
        self.__istop = False
        self.__max_iter = max_iter
        self.__gamma = gamma
        self.__tab_utility: list = []
        self.__tab_frequency: list = []
        self.__tab_visited_state: list = []

    def __alpha(self, n: int) -> float:
        return 180 / (179 + n)

    def __update_utility(self, s_prime, r: int) -> None:  # U[s] + alpha(Ns[s]) (R[s] + γU[s′] − U[s])
        if s_prime not in self.__tab_visited_state:
            self.__tab_utility.append(0)
            self.__tab_frequency.append(0)
            self.__tab_visited_state.append(s_prime)

        if self.__s is not None:
            index_s = self.__tab_visited_state.index(self.__s)
            index_s_prime = self.__tab_visited_state.index(s_prime)
            self.__tab_frequency[index_s] = self.__tab_frequency[index_s] + 1
            self.__tab_utility[index_s] += self.__alpha(self.__tab_frequency[index_s]) *\
                (r + self.__gamma * self.__tab_utility[index_s_prime] \
                 - self.__tab_utility[index_s])

    def __up_and_right_policy(self):
        if self.__istop:
            return 2  # right
        else:
            return 1  # up

    def __debug_env(self, env, iteration: int, s_prime=None, r=None, done_stage=None):
        env.render()
        print("Iteration : "+str(iteration))
        print("s: " + str(self.__s))
        print("s_prime: " + str(s_prime))
        print("reward: " + str(r))
        print("done: " + str(done_stage))
        print("U[s] = "+str(self.__tab_utility))
        print("state[s] = "+str(self.__tab_visited_state))
        print("N[s] = "+str(self.__tab_frequency))
        print("\n")

    def up_and_right_learning(self, env):
        if self.__debug:
            self.__debug_env(env, -1)
        for i in range(self.__max_iter):
            a = self.__up_and_right_policy()
            s_prime, r, done_stage = env.step(a)
            if s_prime == self.__s:
                self.__istop = True
            self.__update_utility(s_prime, r)
            self.__s = s_prime

            if self.__debug:
                self.__debug_env(env, i, s_prime, r, done_stage)

            if done_stage:
                self.__s = None
                self.__istop = False
                env.reset()
                if self.__debug:
                    self.__debug_env(env, i, s_prime, r, done_stage)


def main():
    print("Hello World!")


if __name__ == '__main__':
    main()
