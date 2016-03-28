__author__ = 'jiangdon'

from sklearn.neighbors import KNeighborsClassifier


class KNearestNeighbor:
    def __init__(self):
        pass

    def __init__(self, dummy_x, dummy_y):
        self.dummy_x = dummy_x
        self.dummy_y = dummy_y

    def train(self):
        knn = KNeighborsClassifier()
        knn.fit(self.dummy_x, self.dummy_y)
        return knn
