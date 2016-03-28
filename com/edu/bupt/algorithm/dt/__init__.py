__author__ = 'jiangdon'

from com.edu.bupt.algorithm.dt.DecisionTree import *


dt = DecisionTree(r'C:\Users\jiangdon\PycharmProjects\MLLearning\datasets\dt.csv')
dummy_x, dummy_y = dt.data2Vector()
# print dummy_x
clf = dt.DecisionTreeModel(dummy_x, dummy_y)
# print clf

newRowX = dummy_x[0, :]
# print newRowX
newRowX[0] = 1
newRowX[2] = 0
# print newRowX
y = clf.predict(newRowX)



