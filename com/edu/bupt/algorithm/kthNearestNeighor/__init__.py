__author__ = 'jiangdon'

from sklearn.datasets import load_iris
from com.edu.bupt.algorithm.kthNearestNeighor.KNearestNeighbor import *

datasets = load_iris()
dummy_x, dummy_y = datasets.data, datasets.target

# print dummy_x[0, :], dummy_y[0]
knn = KNearestNeighbor(dummy_x, dummy_y)
clf = knn.train()
y = clf.predict([5.1, 3.5, 1.4, 0.2])
print y
