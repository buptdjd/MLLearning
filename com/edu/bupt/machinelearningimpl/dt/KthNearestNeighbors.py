__author__ = 'jiangdon'

import csv
import random
from math import *
import operator

class KthNearestNeighbors:
    def __init__(self):
        pass

    # load data from file and split the data to training sets and testing sets
    def load_datasets(self, filename, split):
        with open(filename, 'rb') as csv_file:
            lines = csv.reader(csv_file)
            dataset = list(lines)
            self.train_sets = []
            self.test_sets = []
            for i in range(len(dataset)-1):
                for j in range(4):
                    dataset[i][j] = float(dataset[i][j])
                    if random.random() < split:
                        self.train_sets.append(dataset[i])
                    else:
                        self.test_sets.append(dataset[i])

    # calculate the euclidean distance between instance1 and instance2
    def KthNearestNeighbors_euclideanDistance(self, instance1, instance2, length):
        distance = 0
        for i in range(length):
            distance += pow(instance1[i]-instance2[i], 2)
        return sqrt(distance)

    # get the k neighbors of instance x
    def train(self, x, k):
        x_neighbors = []
        length = len(x)-1
        for i in range(len(self.train_sets)):
            dist = self.KthNearestNeighbors_euclideanDistance(x, self.train_sets[i], length)
            x_neighbors.append((self.train_sets[i], dist))
        x_neighbors.sort(key=operator.itemgetter(1))

        neighbors = []
        for i in range(k):
            neighbors.append(x_neighbors[i][0])
        return neighbors

    # predict the category of x
    def predict(self, neighbors):
        votes = {}
        for i in range(len(neighbors)):
            res = neighbors[i][-1]
            if res in votes:
                votes[res] += 1
            else:
                votes[res] = 1
        sorted_votes = sorted(votes.iteritems(), key=operator.itemgetter(1), reverse=True)
        return sorted_votes[0][0]

    # the accuracy of prediction
    def accuracy(self, predictions):
        correct = 0
        for i in range(len(predictions)):
            if predictions[i] == self.test_sets[i][-1]:
                correct += 1
        return correct/float(len(predictions))*100.0
