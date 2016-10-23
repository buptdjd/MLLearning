__author__ = 'jiangdon'


from com.edu.bupt.machinelearningimpl.knn.KthNearestNeighbors import *


knn = KthNearestNeighbors()
split = 0.67
knn.load_datasets(filename=r'C:\Users\jiangdon\PycharmProjects\MLLearning\datasets\iris.data', split=split)

predictions = []
k = 3
test_sets = knn.test_sets

for i in range(len(test_sets)):
    neighbors = knn.train(test_sets[i], k)
    predictions.append(knn.predict(neighbors))

print knn.accuracy(predictions)

