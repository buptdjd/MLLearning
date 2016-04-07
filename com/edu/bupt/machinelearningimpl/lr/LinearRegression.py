__author__ = 'jiangdon'

import numpy as np


class LinearRegression:
    def __init__(self):
        pass

    def __init__(self, x, y, theta):
        self.x = x
        self.y = y
        self.theta = theta

    def loss(self):
        m = len(self.y)
        tmp = np.dot(self.x, self.theta)-self.y
        return 1.0/(2*m)*np.dot(tmp.T, tmp)

    def train(self, iterations, alpha):
        m = len(self.y)
        for i in range(iterations):
            self.theta -= alpha*1.0/m*np.dot(self.x.T, np.dot(self.x, self.theta)-self.y)

    def predict(self, x):
        pass