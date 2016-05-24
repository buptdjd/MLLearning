
import numpy as np
from math import fabs

class DFP:

    def __init__(self):
        pass

    def hessian_dfp(self, c, delta_w, delta_g, epsilon):
        n = delta_w.size
        d = delta_g.dot(delta_w)
        if fabs(d) < epsilon:
            return np.identity(n)
        a = np.zeros((n, n))
        a = delta_w.dot(delta_w)
        a /= d
        b = np.zeros((n, n))
        w2 = c.dot(delta_g)
        b = w2.dot(w2)
        d = delta_g.dot(w2)
        if fabs(d) < epsilon:
            return np.identity(n)
        b /= d
        return c+a-b

    def lr_dfp(self, data, alpha, epsilon):
        n = len(data[0]) - 1
        pre_w = np.zeros(n)
        cur_w = np.zeros(n)
        pre_g = np.zeros(n)
        cur_g = np.zeros(n)
        c = np.identity(n)
        for t in range(100):
            for d in data:
                x = np.array(d[:-1])
                y = d[-1]
                c = self.hessian_dfp(c, (cur_w-pre_w), (cur_g-pre_g), epsilon)
                pre_w = cur_w
                pre_g = cur_g
                cur_g = (y - np.dot(cur_w, x))*x
                cur_w = cur_w + alpha * c.dot(cur_g)
            print t, cur_w
        return cur_w


if __name__ == '__main__':
    dfp = DFP()
    data = np.genfromtxt(r'/Users/jiangdon/Pycharm/MLLearning/datasets/ex1data2.txt'
                     , delimiter=',')
    alpha = 0.02
    epsilon = 1e-5
    dfp.lr_dfp(data, alpha, epsilon)
