import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize


class LogisticRegression:
    def __init__(self):
        pass

    # @param path: data input path
    # @param delimiter: different columns represent as different features, the delimiter concat different features
    # @return training set (X and y)
    def loadData(self, path, delimiter):
        self.data = np.loadtxt(path, delimiter=delimiter)
        m, n = self.data.shape
        X = np.c_[np.ones((m, 1)), self.data[:, 0:n-1]]
        y = np.c_[self.data[:, n-1]]
        return X, y

    # @param z: X.dot(theta.T) will be z
    # @return: project X into [0, 1]
    def sigmoid(self, z):
        return 1.0/(1.0+np.exp(-z))

    # cost function
    # @param theta: weight of logistic regression
    # @param X: features
    # @param y: label
    # @return cost of logistic regression with new weights
    def func_cost(self, theta, X, y):
        m = y.size
        h = self.sigmoid(X.dot(theta))
        J = -1.0*(1.0/m)*(np.log(h).T.dot(y) + np.log(1-h).T.dot(1-y))
        if np.isnan(J[0]):
            return np.inf
        return J[0]

    # @param theta: weight of logistic regression
    # @param X: features
    # @param y: label
    # @return gradient of logistic regression
    def gradient(self, theta, X, y):
        m = y.size
        h = self.sigmoid(X.dot(theta.reshape(-1, 1)))
        grad = (1.0/m)*(X.T.dot(h-y))
        return grad.flatten()

    # @param X: features
    # @param y: label
    # @return the logistic regression model
    def train(self, X, y):
        initial_theta = np.zeros(X.shape[1])
        res = minimize(self.func_cost, initial_theta, args=(X, y), jac=self.gradient, options={'maxiter':400})
        return res

    # @param theta: weight of logistic regression
    # @param X: features
    # @param threshold: the threshold to classify, if h > threshold, we label it 1, and if h<threshold, we label it 0
    def predict(self, theta, X, threshold=0.5):
        h = self.sigmoid(X.dot(theta.T))
        p = h > threshold
        return p.astype('int')

    # draw different label data in 2-D picture
    # @param data: data set
    def draw(self, data):
        X = data[:, 0:2]
        y = data[:, 2]
        pos = np.where(y == 1)
        neg = np.where(y == 0)
        plt.scatter(X[pos, 0], X[pos, 1], marker='o', c='b')
        plt.scatter(X[neg, 0], X[neg, 1], marker='x', c='r')
        plt.xlabel("chinese")
        plt.ylabel("english")
        plt.legend(['fail', 'pass'])

    # draw decision boundary
    def draw_decision_boundary(self, data, X, theta):
        self.draw(data)
        x1_min, x1_max = X[:, 1].min(), X[:, 1].max()
        x2_min, x2_max = X[:, 2].min(), X[:, 2].max()
        x1_range, x2_range = np.meshgrid(np.linspace(x1_min, x1_max), np.linspace(x2_min, x2_max))
        h = self.sigmoid(np.c_[np.ones((x1_range.ravel().shape[0], 1)), x1_range.ravel(), x2_range.ravel()].dot(theta))
        h = h.reshape(x1_range.shape)
        plt.contour(x1_range, x2_range, h, [0.5], linewiths=1, colors='b')
        plt.show()

if __name__ == '__main__':
    lr = LogisticRegression()
    path = "D:\\github\\MLLearning\\datasets\\scoredata1.txt"
    delimiter = ','
    X, y = lr.loadData(path, delimiter)
    model = lr.train(X, y)
    lr.draw_decision_boundary(lr.data, X, model.x)
    print lr.predict(np.array([1, 65, 85]), model.x)