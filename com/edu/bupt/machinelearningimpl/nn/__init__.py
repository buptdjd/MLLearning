__author__ = 'jiangdon'


import numpy as np
from com.edu.bupt.machinelearningimpl.nn.NeuralNetwork import *

# print 2*np.random.random((2, 3))-1

# print np.ones([2,3])

# X = np.atleast_2d([[1, 2], [2, 3]])
# print [X[0]]

# print np.dot(2, [2, 3])

# print (2*np.random.random((2, 3))-1)*0.25

# X = [[1, 2], [3, 4]]
# weights = []
# layers = [2, 3, 2]
# for i in range(1, len(layers)-1):
#     weights.append((2*np.random.random((layers[i-1]+1, layers[i]+1))-1)*0.25)
#     weights.append((2*np.random.random((layers[i]+1, layers[i+1]))-1)*0.25)
#
# X = np.atleast_2d(X)
# temp = np.ones([2, 3])
# temp[:, 0:-1] = X
# X = temp
# a = [X[0]]
#
# for i in range(len(weights)):
#     print np.dot(a[i], weights[i]), 'end'
#     a.append(np.tanh(np.dot(a[i], weights[i])))
# print a

# a = np.array([1, 2, 3])
# b = np.array([4, 5, 6])
# print a*b

X = [[0, 0], [0, 1], [1, 0], [1, 1]]
y = [0, 1, 1, 0]
layers = [2, 2, 1]
neuralnetwork = NeuralNetwork(layers, activation='tanh')
neuralnetwork.fit(X, y)
x = [0, 0]
print neuralnetwork.predict(x)

