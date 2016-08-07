from math import log
from operator import itemgetter
from com.edu.bupt.utils.FileUtils import *


class ID3DTree:
    def __init__(self):
        pass

    def __init__(self, data):
        self.data = data

    '''
    this method is used to calculate the entropy of data
    :param data
            the input sample like (f1, f2, f3, ..., y1)
    :return
            entropy of data
    '''

    @staticmethod
    def cal_shannon_entropy(data):
        # the size of training set
        numbers = len(data)
        label_count = {}
        for vec in data:
            cur_label = vec[-1]
            if cur_label not in label_count.keys():
                label_count[cur_label] = 1
            else:
                label_count[cur_label] += 1
        shannon_entropy = 0.0
        for key in label_count:
            prob = float(label_count[key])/numbers
            shannon_entropy -= prob * log(prob, 2)
        return shannon_entropy

    '''
    this method is used for splitting the data set with specific feature
    :param data
            the data set which we input
    :param axis
            the feature index in data set
    :param value
            the feature value in data set
    :return
            data set which is split with the specific feature
    '''
    def split_datasets(self, data, axis, value):
        ret = []
        for vec in data:
            if vec[axis] == value:
                reduced_vector = vec[:axis]
                reduced_vector.extend(vec[axis+1:])
                ret.append(reduced_vector)
        return ret

    '''
    this method is used for choosing the best feature to split the data set
    :param data
             the data set which we input
    :return
            the best feature
    '''
    def choose_best_feature_split(self, data):
        feature_num = len(data[0])-1
        base_entropy = self.cal_shannon_entropy(data)
        best_info_gain = 0
        best_feature = -1
        for i in range(feature_num):
            feature_set = set([example[i] for example in data])
            new_entropy = 0.0
            for value in feature_set:
                sub_data = self.split_datasets(data, i, value)
                prob = len(sub_data)/float(len(data))
                new_entropy += prob * self.cal_shannon_entropy(sub_data)
            info_gain = base_entropy - new_entropy
            if info_gain > best_info_gain:
                best_info_gain = info_gain
                best_feature = i
        return best_feature


    '''
    this method is used for picking up the category when data set is consist of only one feature
    :param class_list
        labels for data set
    :return
        the category
    '''
    @staticmethod
    def majority_cnt(class_list):
        class_count = {}
        for vote in class_list:
            if vote not in class_count.keys():
                class_count[vote] = 0
            class_count[vote] += 1

        sorted_class_count = sorted(class_count.iteritems(), key=itemgetter(1), reverse=True)
        return sorted_class_count[0][0]

    '''
    this method is used for creating decision tree
    :param
        data set
    :return
        decision tree
    '''
    def create_tree(self, data, labels):
        class_list = [example[-1] for example in data]
        # the type is the same, so stop classify
        if class_list.count(class_list[0]) == len(class_list):
            return class_list[0]
        # traversal all the features and choose the most frequent feature
        if len(data[0]) == 1:
            return self.majority_cnt(class_list)

        best_feature = self.choose_best_feature_split(data)
        best_feature_label = labels[best_feature]
        tree = {best_feature_label: {}}
        # del(labels[best_feature])
        # get the list which attain the whole properties
        feature_values = [example[best_feature] for example in data]
        unique_values = set(feature_values)
        for value in unique_values:
            sub_labels = labels[0:best_feature]
            sub_labels.extend(labels[best_feature+1:])
            tree[best_feature_label][value] = self.create_tree(
                self.split_datasets(data, best_feature, value), sub_labels)
        return tree

    '''
    this method is used for predicting
    :param tree_model
            training model
    :param feature_labels
            the label of features
    :param test
            the testing data
    :return
            category of data
    '''
    def predict(self, tree_model, feature_labels, test):
        first_str = tree_model.keys()[0]
        second_dict = tree_model[first_str]
        feature_ind = feature_labels.index(first_str)
        for key in second_dict.keys():
            if test[feature_ind] == key:
                if type(second_dict[key]).__name__ == 'dict':
                    label = self.predict(second_dict[key], feature_labels, test)
                else:
                    label = second_dict[key]
        return label


def create_data(path):
    file_utils = FileUtils()
    reader = file_utils.getCsvReader(path)
    data = []
    for row in reader:
        sub = []
        for i in range(0, len(row)-1):
            sub.append(float(row[i]))
        sub.append(row[len(row)-1])
        data.append(sub)
    return data

if __name__ == "__main__":
    '''
    data = [[1, 1, 'yes'],
               [1, 0, 'no'],
               [0, 1, 'no'],
               [0, 1, 'no']]
    label_set = ['no surfacing', 'flippers']
    '''
    path = r"D:\Users\Michael\PycharmProjects\MLLearning\datasets\iris.data"
    data_set = create_data(path)
    feature_label = ['sepal length', 'sepal width', 'petal length', 'petal width']
    tree = ID3DTree(data_set)
    myTree = tree.create_tree(data_set, feature_label)
    print myTree
    print tree.predict(myTree, feature_label, [6.5, 2.9, 5.2, 2.0])

