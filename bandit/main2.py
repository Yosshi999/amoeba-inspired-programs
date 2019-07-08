import numpy as np
import matplotlib.pyplot as plt
from main import *
import pandas as pd
from algorithm import *


"""
https://blog.albert2005.co.jp/2017/01/23/%E3%83%90%E3%83%B3%E3%83%87%E3%82%A3%E3%83%83%E3%83%88%E3%82%A2%E3%83%AB%E3%82%B4%E3%83%AA%E3%82%BA%E3%83%A0%E3%80%80%E5%9F%BA%E6%9C%AC%E7%B7%A8/

modified by Yosshi999
"""
def test_algorithm(algo, arms, num_sims, horizon, xa, xb, s, q):
    chosen_arms = np.zeros(num_sims * horizon)
    cumulative_rewards = np.zeros(num_sims * horizon)
    times = np.zeros(num_sims * horizon)
    for sim in range(num_sims):
        algo.initialize(len(arms))
        for t in range(horizon):
            index = sim * horizon + t
            times[index] = t + 1
            chosen_arm = algo.select_arm()
            chosen_arms[index] = chosen_arm
            reward = arms[chosen_arm].draw()
            if t == 0:
                cumulative_rewards[index] = reward
            else:
                cumulative_rewards[index] = cumulative_rewards[
                    index - 1] + reward
            algo.update(chosen_arm, reward)

            if sim==0 and algo.__class__.__name__ == "ToWSampling":
                xa.append(algo.model.x[0])
                xb.append(algo.model.x[1])
                s.append(algo.model.s)
                q.append(algo.model.q)
    return [times, chosen_arms, cumulative_rewards]

############################################################

class ModEpsilonGreedy(EpsilonGreedy): 
    def __init__(self, tau, counts, values):
        eps = 1/(1+tau*0)
        super().__init__(eps, counts, values)
        self.step = 0
        self.tau = tau
 
    def initialize(self, n_arms):
        self.step = 0
        self.epsilon = 1/(1+self.tau*0)
        self.counts = np.zeros(n_arms)
        self.values = np.zeros(n_arms)
 
 
    def update(self, chosen_arm, reward):
        self.step += 1
        self.epsilon = 1/(1 + self.tau * self.step)
        super().update(chosen_arm, reward)
 



class ToWSampling:
    def __init__(self, alpha=1, mu=1):
        self.model = ToW(alpha, mu)
        self.buffer = []
        self.choice = np.zeros(2)
        self.light = np.zeros(2)
    def initialize(self, n_arms):
        assert n_arms == 2
        self.buffer = []
        self.model.initialize()
        self.choice[:] = 0
        self.light[:] = 0 
    def select_arm(self):
        if len(self.buffer) == 0:
            while True:
                act = self.model.act()
                if np.sum(act) > 0: break
                self.model.update(np.zeros(2), np.zeros(2))
            self.choice[:] = 0
            self.choice[act] = 1
            self.light[:] = 0
            self.buffer.extend(np.where(act)[0].tolist())
            assert len(self.buffer) > 0
        #print(self.buffer)
        a = self.buffer.pop()
        return a
    def update(self, chosen_arm, reward):
        if reward != 1:
            assert self.choice[chosen_arm] == 1
            self.light[chosen_arm] = 1
        if len(self.buffer) == 0:
            self.model.update(self.choice, self.light)
            
        

algos = [
    #ToWSampling(mu=5.),
    ToWSampling(alpha=1, mu=3),
    EpsilonGreedy(0.2, [], []),
    ModEpsilonGreedy(0.05, [], []),
    #ModEpsilonGreedy(0.4, [], []),
    UCB([], []),
    ThompsonSampling([], [], []),
]

def test():
    xa, xb, s, q = [], [], [], []
    p = [0.45, 0.55]
    arms = list(map(lambda x: BernoulliArm(x), p))
    nsim = 100
    ntry = 800

    random.seed(0)
    np.random.seed(0)

    plt.figure(figsize=(8,4))
    plt.suptitle("P=" + str(p))
    plt.subplot(1,2,1)

    for al in algos:
        print(al.__class__.__name__)
        al.initialize(2)
        result = test_algorithm(al, arms, nsim, ntry, xa, xb, s, q)
        times = result[0].reshape(nsim, ntry)[0]
        best_arms = (result[1] == np.argmax(p)).astype(np.int).reshape(nsim, ntry).mean(axis=0)
        best_arms_n = np.add.accumulate(best_arms)
        best_arms_rate = best_arms_n / times
        plt.plot(times, best_arms_rate, label=al.__class__.__name__,
            lw = 2 if al.__class__.__name__ == "ToWSampling" else 1)

    plt.ylim(0.4, 1.0)
    plt.legend()

    plt.subplot(1,2,2)

    plt.plot(np.arange(ntry), xa, label="Xa")
    plt.plot(np.arange(ntry), xb, label="Xb")
    plt.plot(np.arange(ntry), s, label="S")
    plt.plot(np.arange(ntry), q, label="Q")
    plt.legend()
    plt.show()

def test2():
    xa, xb, s, q = [], [], [], []
    p = [0.2, 0.8]
    arms = list(map(lambda x: BernoulliArm(x), p))
    nsim = 100
    ntry = 800

    random.seed(0)
    np.random.seed(0)

    plt.figure(figsize=(8,4))
    plt.suptitle("P=" + str(p))
    plt.subplot(1,2,1)

    for al in algos:
        print(al.__class__.__name__)
        al.initialize(2)
        result = test_algorithm(al, arms, nsim, ntry, xa, xb, s, q)
        times = result[0].reshape(nsim, ntry)[0]
        best_arms = (result[1] == np.argmax(p)).astype(np.int).reshape(nsim, ntry).mean(axis=0)
        best_arms_n = np.add.accumulate(best_arms)
        best_arms_rate = best_arms_n / times
        plt.plot(times, best_arms_rate, label=al.__class__.__name__,
            lw = 2 if al.__class__.__name__ == "ToWSampling" else 1)

    plt.ylim(0.92, 1.0)
    plt.legend()

    plt.subplot(1,2,2)

    plt.plot(np.arange(ntry), xa, label="Xa")
    plt.plot(np.arange(ntry), xb, label="Xb")
    plt.plot(np.arange(ntry), s, label="S")
    plt.plot(np.arange(ntry), q, label="Q")
    plt.legend()
    plt.show()

if __name__ == '__main__':
    test()
    test2()
