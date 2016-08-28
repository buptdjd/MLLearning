import numpy as np
import random
from scipy.linalg import norm
import numpy.matlib
import matplotlib.pyplot as plt
import cPickle as pickle


class KMeans:
    def __init__(self):
        pass

    # param: X training data
    # param: k clusters
    # param: observer function to draw
    # param: threshold
    # param: max_iter maximum iterations
    def __init__(self, X, k, observer=None, threshold=1e-15, max_iter=300):
        n = len(X)
        self.X = X
        self.k = k
        self.observer = observer
        self.threshold = threshold
        self.max_iter = max_iter
        self.labels = np.zeros(n, dtype=int)
        self.centers = np.array(random.sample(X, k))

    # param: X training data
    # param: centers cluster point
    # param: labels label for training data
    # return: cost function for k-means
    def cal_J(self, X, centers, labels):
        n = len(X)
        _sum = 0
        for i in xrange(n):
            _sum += norm(X[i]-centers[labels[i]])
        return _sum

    # param: X training data
    # param: Y cluster
    # return: the distance between training data and clusters
    # we can convert k-means to operation of matrix
    def distmat(self, X, Y):
        n = len(X)
        m = len(Y)
        xx = numpy.matlib.sum(X*X, axis=1)
        yy = numpy.matlib.sum(Y*Y, axis=1)
        xy = numpy.matlib.dot(X, Y.T)
        return np.tile(xx, (m, 1)).T+np.tile(yy, (n, 1))-2*xy

    # training k-means
    def train(self):
        J_pre = self.cal_J(self.X, self.centers, self.labels)
        iter = 0
        while True:
            if self.observer is not None:
                self.observer(self.X, iter, self.labels, self.centers)
            dist = self.distmat(self.X, self.centers)
            labels = dist.argmin(axis=1)
            for j in range(self.k):
                idx_j = (labels == j).nonzero()
                self.centers[j] = self.X[idx_j].mean(axis=0)
            J = self.cal_J(self.X, self.centers, self.labels)
            iter += 1
            if abs(J_pre-J) < self.threshold:
                break
            J_pre = J
            if iter > self.max_iter:
                break
        if self.observer is not None:
            self.observer(self.X, iter, self.labels, self.centers)

if __name__ == '__main__':
    # load previously generated points
    path = "D:\Users\Michael\PycharmProjects\MLLearning\datasets\cluster.pkl"
    with open(path) as inf:
        samples = pickle.load(inf)
    N = 0
    for smp in samples:
        N += len(smp[0])
    X = np.zeros((N, 2))
    idxfrm = 0
    for i in range(len(samples)):
        idxto = idxfrm + len(samples[i][0])
        X[idxfrm:idxto, 0] = samples[i][0]
        X[idxfrm:idxto, 1] = samples[i][1]
        idxfrm = idxto

    def observer(X, iter, labels, centers):
        print "iter %d." % iter
        colors = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        plt.plot(hold=False)  # clear previous plot
        plt.hold(True)

        # draw points
        data_colors = [colors[lbl] for lbl in labels]
        plt.scatter(X[:, 0], X[:, 1], c=data_colors, alpha=0.5)
        # draw centers
        plt.scatter(centers[:, 0], centers[:, 1], s=200, c=colors)
        save_path = "D:\Users\Michael\PycharmProjects\MLLearning\ouput\kmeans\iter_%02d.png"
        plt.savefig(save_path % iter, format='png')

    kmeans = KMeans(X, 3, observer=observer)
    kmeans.train()



