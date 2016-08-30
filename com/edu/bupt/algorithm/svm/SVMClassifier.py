
from sklearn import svm
import numpy as np
import pylab as pl


class SVMClassifier:
    def __init__(self):
        pass

    def __init__(self, X, y):
        self.X = X
        self.y = y

    def train(self):
        # use linear kernel to generate svm classifier
        clf = svm.SVC(kernel='linear')
        clf.fit(self.X, self.y)
        return clf


# generate X,y for training set
def data_generated():
    np.random.seed(0)
    # np.r_ can convert [[[2, 1], [3, 4]], [[3, 4], [2, 1]]] to [[2, 1], [3, 4], [3, 4], [2, 1]]
    X = np.r_[np.random.randn(20, 2) - [2, 2], np.random.randn(20, 2) + [2, 2]]
    y = [0]*20 + [1]*20
    return X, y


# draw figures for svm classifier
def svm_figure_generate(w, b, support_vectors, X):
    k = - w[0]/w[1]
    x = np.linspace(-5, 5)
    y = k*x - b/w[1]
    sv_1 = support_vectors[0]
    yy_down = k*x + (sv_1[1]-k*sv_1[0])
    sv_2 = support_vectors[-1]
    yy_up = k*x + (sv_2[1] - k*sv_2[0])
    pl.plot(x, y, 'k-')
    pl.plot(x, yy_up, 'k--')
    pl.plot(x, yy_down, 'k--')
    pl.scatter(support_vectors[:, 0], support_vectors[:, 1], s=80, facecolor='none')
    pl.scatter(X[:, 0], X[:, 1], c='Y', cmap=pl.cm.Paired)
    pl.axis('tight')
    pl.show()

if __name__ == '__main__':
    X, y = data_generated()
    model = SVMClassifier(X, y)
    clf = model.train()
    svm_figure_generate(clf.coef_[0], clf.intercept_, clf.support_vectors_, X)

