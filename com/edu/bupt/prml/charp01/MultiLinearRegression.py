__author__ = 'jiangdon'

import numpy as np
import matplotlib.pyplot as plt


class MultiLinearRegression:
    def __init__(self):
        pass

    def __init__(self, x, y, theta):
        self.x = x
        self.y = y
        self.theta = theta

    # normalize the features
    def feature_normalize(self):
        x_norm = self.x
        # x is a n*m matrix, np.mean(x, axis) will compute features mean value,
        # axis=0 represent the column mean
        # axis=1 represent the row mean
        mu = np.mean(self.x, axis=0) # mean
        sigma = np.std(self.x, axis=0) # standard deviation
        for i in range(len(mu)):
            x_norm[:, i] = (self.x[:, i]-mu[i])/sigma[i]
        return x_norm, mu, sigma

    @classmethod
    def compute_cost(self, x, y, theta):
        m = len(y)
        tmp = np.dot(x, theta) - y
        cost = 1.0/(2*m)*np.dot(tmp.T, tmp)
        return cost

    def gradient_descent(self, iterations=10000, alpha=0.2):
        m = len(self.y)
        j_history = []
        # training
        for i in range(iterations):
            self.theta -= alpha * (1.0/m)*np.dot(self.x.T, np.dot(self.x, self.theta)-self.y)
            j_history.append(self.compute_cost(self.x, self.y, self.theta))

        return self.theta, j_history

    def normal_equ(self):
        theta = np.linalg.inv(np.dot(self.x.T, self.x))
        theta = np.dot(theta, self.x.T)
        theta = np.dot(theta, self.y)
        return theta

    

if __name__ == "__main__":
    path = "/Users/jiangdon/Pycharm/MLLearning/datasets/ex1data2.txt"
    data = np.genfromtxt(path, delimiter=",")
    x = data[:,(0,1)]
    y = data[:,2]
    theta = np.zeros(3)
    mlr = MultiLinearRegression(x, y, theta)
    x, mu, sigma = mlr.feature_normalize()

    m = len(y)
    # add the bias
    x = np.hstack((np.ones((m,1)), x))

    iterations = 1500
    alpha = 0.01

    initialCost = mlr.compute_cost(x, y, theta)
    print initialCost

    mlr.x = x
    theta, j_history = mlr.gradient_descent(iterations, alpha)
    print theta, j_history[-1]

    weight = mlr.normal_equ()
    print weight

    plt.figure(1)
    plt.plot(j_history)
    plt.xlabel("iterations")
    plt.ylabel("cost")
    plt.show()
