__author__ = 'jiangdon'

from com.edu.bupt.machinelearningimpl.lr.LinearRegression import *
import numpy as np

data = np.genfromtxt(r'C:\Users\jiangdon\PycharmProjects\MLLearning\datasets\ex1data1.txt'
                     , delimiter=',')

x = data[:, 0]
y = data[:, -1]
m = len(y)
x = x.reshape((m, 1))
x = np.hstack((np.ones((m, 1)), x))
theta = np.zeros(2)
lr = LinearRegression(x, y, theta)
lr.train(iterations=100, alpha=0.01)
print lr.theta