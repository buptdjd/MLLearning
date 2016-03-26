__author__ = 'jiangdon'

from com.edu.bupt.algorithm.dt.DecisionTree import *


dt = DecisionTree(r'C:\Users\jiangdon\PycharmProjects\MLLearning\datasets\dt.csv')
dummy_x, dummy_y = dt.data2Vector()
clf = dt.DecisionTreeModel(dummy_x, dummy_y)
print clf



