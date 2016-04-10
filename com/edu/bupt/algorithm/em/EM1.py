
import math
import copy
import numpy as np
import matplotlib.pyplot as plt

# there is a scenario that students play on campus, we get the statistics about students' heights
# boys' height will be a normal distribution (mu1, sigma)
# girls' height will be a normal distribution (mu2, sigma)
# we only get the height samples and want to estimate the mu1 and mu2
# em algorithm can solve the problem, so we use the em tools to do

class EMAlgor:
    def __init__(self):
        pass

    def __init__(self, x, expectations, mu):
        self.x = x
        self.expectations = expectations
        self.mu = mu

    # the expectations of em
    def e_step(self, sigma, k, n):
        for i in xrange(0, n):
            sums = 0
            for j in xrange(0, k):
                sums += math.exp(-float(1)/float(2*sigma**2)*(self.x[i, :]-self.mu[j])**2)
            for j in xrange(0, k):
                item = math.exp(-float(1)/float(2*sigma**2)*(self.x[i, :]-self.mu[j])**2)
                self.expectations[i, j] = item/sums

    # the maximum of em
    def m_step(self, k, n):
        for j in xrange(0, k):
            item = 0
            sums = 0
            for i in xrange(0, n):
                sums += self.expectations[i, j]*self.x[i, 0]
                item += self.expectations[i, j]
            self.mu[j] = sums/item

    def em(self, iterations, epsilon, sigma, k, n):
        for i in range(iterations):
            old_mu = copy.deepcopy(self.mu)
            self.e_step(sigma, k, n)
            self.m_step(k, n)
            print i, self.mu
            if sum(abs(self.mu - old_mu)) < epsilon:
                break

if __name__ == '__main__':
    n = 1000
    k = 2
    x = np.zeros((n, 1))
    sigma = 6
    mu1 = 40
    mu2 = 20
    mu = np.random.random(k)
    expectations = np.zeros((n, k))
    iterations = 1000
    epsilon = 0.0001
    for i in xrange(0, n):
        if np.random.random(1)> 0.5:
            x[i, 0] = np.random.normal()*sigma + mu1
        else:
            x[i, 0] = np.random.normal()*sigma + mu2
    em = EMAlgor(x, expectations, mu)
    em.em(iterations, epsilon, sigma, k, n)
    plt.hist(x, 50)
    plt.show()
