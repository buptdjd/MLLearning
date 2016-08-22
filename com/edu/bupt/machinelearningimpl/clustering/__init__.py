import numpy as np
import numpy.matlib as ml


def distmat(X, Y):
    n = len(X)
    m = len(Y)
    xx = ml.sum(X * X, axis=1)
    yy = ml.sum(Y * Y, axis=1)
    xy = ml.dot(X, Y.T)
    print np.tile(xx, (m, 1))
    print np.tile(yy, (n, 1))
    return np.tile(xx, (m, 1)).T + np.tile(yy, (n, 1)) - 2 * xy

x = np.array([[1, 2, 3], [4, 5, 6]])
y = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]])


print distmat(x, y)