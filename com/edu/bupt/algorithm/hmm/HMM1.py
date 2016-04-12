

class HMM1:
    def __init__(self):
        pass

    def __init__(self, states, obs,  p_trans, p_emit, pi):
        self.states = states
        self.obs = obs
        self.p_trans = p_trans
        self.p_emit = p_emit
        self.pi = pi


    def viterbi(self):
        delta = [{}]
        path = {}
        # t=0 and initialize the v
        for y in self.states:
            delta[0][y] = self.pi[y]*self.p_emit[y][self.obs[0]]
            path[y] = [y]

        T = len(self.obs)
        for t in range(1, T):
            delta.append({})
            new_path = {}
            for y in self.states:
                prob = -1
                state = self.states[0]
                # max delta[j]*a[j,i]*b[i,o(t+1)]
                for y0 in self.states:
                    p = delta[t-1][y0]*self.p_trans[y0][y]*self.p_emit[y][self.obs[t]]
                    if prob < p:
                        prob = p
                        state = y0
                delta[t][y] = prob
                # record the state
                new_path[y] = path[state] + [y]
            path = new_path
        prob = -1
        state = self.states[0]
        for y in self.states:
            if prob < delta[T-1][y]:
                prob = delta[T-1][y]
                state = y

        return prob, path[state]


if __name__ == '__main__':
    states = ('Rainy', 'Sunny')
    obs = ('walk', 'shop', 'clean')
    pi = {'Rainy': 0.6, 'Sunny': 0.4}
    p_trans = {
        'Rainy' : {'Rainy': 0.7, 'Sunny': 0.3},
        'Sunny' : {'Rainy': 0.4, 'Sunny': 0.6},
    }
    p_emit= {
        'Rainy' : {'walk': 0.1, 'shop': 0.4, 'clean': 0.5},
        'Sunny' : {'walk': 0.6, 'shop': 0.3, 'clean': 0.1},
    }

    hmm = HMM1(states, obs, p_trans, p_emit, pi)
    prob, states = hmm.viterbi()
    print prob, states

