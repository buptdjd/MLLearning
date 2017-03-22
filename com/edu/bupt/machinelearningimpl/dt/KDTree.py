import numpy as np
import math


class TreeNode:
	def __init__(self, point=None, split=None, left=None, right=None):
		self.point = point
		self.split = split
		self.left = left
		self.right = right


# construct a KD tree from a m * n data matrix
def createKDTree(data_list):
	# sample size of data_list
	_size = len(data_list)
	if _size==0:
		return
	# feature dimension of data_list
	dimension = len(data_list[0])
	max_var = 0
	split = 0
	# choose feature and the strategy is to minimize the variance of different features
	# compute the variance of data list with specific feature
	for i in range(dimension):
		l1 = []
		for line in data_list:
			l1.append(line[i])
		var = compute_variance(l1)
		if var > max_var:
			max_var = var
			split = i
	# sort data list in according to value of the specific feature
	data_list.sort(key=lambda x: x[split])
	# root node is the median of data_list
	point = data_list[_size/2]
	root = TreeNode(point, split)
	root.left = createKDTree(data_list[0:(_size/2)])
	root.right = createKDTree(data_list[(_size/2+1):-1])
	return root


# compute variance of l1 which is consist of one feature
def compute_variance(l1):
	for f in l1:
		f = float(f)
	_size = len(l1)
	_arr1 = np.array(l1)
	_sum1 = _arr1.sum()
	_arr2 = _arr1 * _arr1
	_sum2 = _arr2.sum()
	_mean = _sum1/_size
	# D(X) = E(X^2) - (E(x))^2
	_var = _sum2/_size - _mean**2
	return _var


# compute the Euclid distance between p1 and p2
def computeDist(p1, p2):
	_sum = 0.0
	for i in range(len(p1)):
		_sum += (p1[i]-p2[i])**2
	return math.sqrt(_sum)


# find the nearest point of query node
def findNN(root, query):
	NN = root.point
	min_dist = computeDist(query, NN)
	node_list = []
	temp = root
	# find the nearest node of query node
	while temp:
		node_list.append(temp)
		dd = computeDist(query, temp.point)
		if min_dist > dd:
			NN = temp.point
			min_dist = dd
		i = temp.split
		# search left tree
		if query[i] <= temp.point[i]:
			temp = temp.left
		# search right tree
		else:
			temp = temp.right
	# backtrace
	while node_list:
		back_point = node_list.pop()
		i = back_point.split
		if abs(query[i]-back_point.point[i]) < min_dist:
			if query[i] <= back_point.point[i]:
				temp = back_point.right
			else:
				temp = back_point.left
		if temp:
			node_list.append(temp)
			cur_dist = computeDist(query, temp.point)
			if min_dist > cur_dist:
				min_dist = cur_dist
				NN = temp.point
	return NN, min_dist


def knn(l1, query):
	min_dist = 999999.0
	NN = l1[0]
	for p in l1:
		dist = computeDist(query, p)
		if dist < min_dist:
			min_dist = dist
			NN = p
	return NN, min_dist

def preorder(root):
	print root.point
	if root.left:
		preorder(root.left)
	if root.right:
		preorder(root.right)


if __name__ == '__main__':
	data = [[1,2], [2, 3], [4, 5], [3,6]]
	n1, d1 = knn(data, [2, 4])
	root = createKDTree(data)
	n2, d2 = findNN(root, [2, 4])
	print n1, d1
	print n2, d2
	preorder(root)

