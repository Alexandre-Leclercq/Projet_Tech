from environment import SimpleMaze


class Agent:
    def __init__(self, gamma: int, debug: bool = False):
        self.__debug: bool = debug
        self.__s = None  # actual state
        self.__istop = False
        self.__max_iter = 1000
        self.__gamma = gamma
        self.__tab_utility: list = []
        self.__tab_frequency: list = []
        self.__tab_rewards: list = []
        self.__tab_visited_state: list = []

    def __alpha(self, n: int) -> float:
        return 60 / (59 + n)

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

    def __debug_env(self, env, s_prime=None, r=None, done_stage=None):
        env.render()
        print("s: " + str(self.__s))
        print("s_prime: " + str(s_prime))
        print("reward: " + str(r))
        print("done: " + str(done_stage))

    def up_and_right_learning(self, env):
        if self.__debug:
            self.__debug_env(env)
        for _ in range(1000):
            a = self.__up_and_right_policy()
            s_prime, r, done_stage = env.step(a)
            if self.__debug:
                self.__debug_env(env, s_prime, r, done_stage)
            if s_prime == self.__s:
                self.__istop = True
            self.__update_utility(s_prime, r)
            self.__s = s_prime
            print(self.__tab_utility)
            print(self.__tab_visited_state)
            print(self.__tab_frequency)

            if done_stage:
                self.__s = None
                self.__istop = False
                env.reset()
                if self.__debug:
                    self.__debug_env(env)


def main():
    print("Hello World!")


if __name__ == '__main__':
    main()
