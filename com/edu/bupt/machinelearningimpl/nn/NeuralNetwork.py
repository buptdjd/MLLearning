__author__ = 'jiangdon'

import numpy as np


# tanh(x) = (np.exp(x)-np.exp(-x))/(np.exp(x)+np.exp(-x))
def tanh(x):
    return np.tanh(x)

# the derivative of tanh, the function will be used in back propagation
def tanh_derivative(x):
    return 1.0-np.tanh(x)*np.tanh(x)

# the logistic function and the range between 0 and 1
def logistic(x):
    return 1/(1+np.exp(-x))


def logistic_derivative(x):
    return logistic(x)*(1.0-logistic(x))


class NeuralNetwork:
    def __init__(self):
        pass

    # layers represent as layer in neural network, [2, 3, 2] means
    # the number of input layer is 2, the number of hidden layer is 3,
    # the number of output layer is 2
    # activation represent as activate function
    def __init__(self, layers, activation='tanh'):
        if activation == 'tanh':
            self.activation = tanh
            self.activation_derivative = tanh_derivative
        elif activation == 'logistic':
            self.activation = logistic
            self.activation_derivative = logistic_derivative

        self.weights = []

        # the initialization of neural network weights
        for i in range(1, len(layers)-1):
            # weight between input layer and hidden layer
            self.weights.append((2*np.random.random((layers[i-1]+1, layers[i]+1))-1)*0.25)
            # weight between hidden layer and output layer
            self.weights.append((2*np.random.random((layers[i]+1, layers[i+1]))-1)*0.25)

    # the training function
    # X and y is train sets, learning_rate is learning rate, epochs is iterations
    def fit(self, X, y, learning_rate=0.2, epochs=10000):
        x = np.atleast_2d(X)
        # add bias to X
        temp = np.ones((x.shape[0], x.shape[1]+1))
        temp[:, 0:-1] = x
        x = temp
        y = np.array(y)
        for k in range(epochs):
            # random to select one sample
            i = np.random.randint(x.shape[0])
            a = [x[i]]

            for l in range(len(self.weights)):
                a.append(self.activation(np.dot(a[l], self.weights[l])))
            error = y[i] - a[-1]
            deltas = [error*self.activation_derivative(a[-1])]

            for l in range(len(a)-2, 0, -1):
                deltas.append(deltas[-1].dot(self.weights[l].T)*self.activation_derivative(a[l]))
            deltas.reverse()
            for i in range(len(self.weights)):
                layer = np.atleast_2d(a[i])
                delta = np.atleast_2d(deltas[i])
                self.weights[i] += learning_rate * layer.T.dot(delta)

    def predict(self, x):
        x = np.array(x)
        temp = np.ones(x.shape[0]+1)
        temp[0:-1] = x
        a = temp
        for l in range(0, len(self.weights)):
            a = self.activation(np.dot(a, self.weights[l]))
        return a