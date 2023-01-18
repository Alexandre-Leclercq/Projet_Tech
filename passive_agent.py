import environment


def main():
    print('Hello world')

class Agent:
    def __init__(self, gamma: int, tab_utility: [], tab_visited_state: [], tab_frequency: [], tab_rewards: []):
        self.__s = None #actual state
        self.__istop = False
        self.__max_iter = 1000
        self.__gamma = gamma
        self.__tab_utility = tab_utility
        self.__tab_frequency = tab_frequency
        self.__tab_rewards = tab_rewards
        self.__tab_visited_state = tab_visited_state

    def alpha(self, n: int):
        return 60/(59+n)

    def update_utility(self,s_prime, r, gamma) -> None : # U[s] + alpha(Ns[s]) (R[s] + γU[s′] − U[s])
        if(s_prime not in self.__tab_visited_state):
            self.__tab_utility.append(0)
            self.__tab_frequency.append(1)
            self.__tab_visited_state.append(s_prime)

        if(self.__s is not None ):
            indexS = self.__tab_visited_state.index(self.__s)
            indexSprime = self.__tab_visited_state.index(s_prime)
            self.__tab_frequency[indexS] = self.__tab_frequency[indexS] + 1
            self.__tab_utility[indexS] += self.alpha(self.__tab_frequency[indexS]) \
                                         (r +self.__gamma * self.__tab_utility[indexSprime] \
                                          - self.__tab_utility[indexS])

    def UpAndRight_policy(self):
        if self.__istop:
            return 2 #droite
        else:
            return 1 #haut

    def upandright_learning(self,s):
        donestage: bool = False
        for _ in range(1000):
            a = self.UpAndRight_policy(self)
            s_prime, r, donestage = environment.step(s, a)
            self.__istop= (True, False)[s_prime == self.__s]
            self.update_utility(self,s_prime,r)

            if donestage:
                environment.reset()





if __name__ == '__main__':
    main()
