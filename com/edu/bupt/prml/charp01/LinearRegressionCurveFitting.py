__author__ = 'jiangdon'

import numpy as np
import matplotlib.pyplot as plt


def plot_data(x, y):
    plt.scatter(x, y, c='red', marker='o', label='training data')
    plt.xlabel('x')
    plt.ylabel('y')


def compute_cost(x, y, theta):
    m = len(y)
    tmp = np.dot(x, theta)-y
    J = 1.0/(2*m)*np.dot(tmp.T, tmp)
    return J


def gradient_descent(x, y, theta, iterations, alpha):
    m = len(y)
    j_history = []
    for i in range(iterations):
        theta -= alpha*1.0/m*np.dot(x.T, np.dot(x, theta)-y)
        j_history.append(compute_cost(x, y, theta))
    return theta, j_history


if __name__ == '__main__':
    x = np.linspace(0, 1, 100)
    y = np.sin(np.pi*2*x) + np.random.normal(0, 0.2, x.size)
    m = len(x)
    plt.figure(1)
    plot_data(x, y)

    x = x.reshape((m, 1))
    x = np.hstack((np.ones((m, 1)), x, x**2, x**3))
    theta = np.zeros(4)
    iterations = 100000
    alpha = 0.2
    initial_cost = compute_cost(x, y, theta)
    print "initial_cost:", initial_cost
    theta, j_history = gradient_descent(x, y ,theta, iterations, alpha)
    print "theta:", theta
    print "j_history:", j_history[-1]
    plt.figure(2)
    plt.plot(j_history)
    plt.xlabel("iterations")
    plt.ylabel("j cost")
    plt.figure(1)
    plt.plot(x[:, 1], np.dot(x, theta), 'b-', label="linear regression")
    plt.xlim((0, 1))
    plt.legend()
    plt.show()

