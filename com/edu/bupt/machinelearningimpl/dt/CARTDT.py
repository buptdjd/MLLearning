from operator import itemgetter


class CARTDTree:
    def __init__(self):
        pass

    '''
    :param data
        data set which will be split
    :param axis
        feature index
    :param value
        split the data set with the help of feature value
    :return
        data set
    '''
    @staticmethod
    def split_data_set(data, axis, value):
        ret = []
        for vec in data:
            if vec[axis] == value:
                part_1 = vec[:axis]
                part_1.extend(vec[axis+1:])
                ret.append(part_1)
        return ret

    '''
    :param data
        data set
    :param flag
        category
    :return
        best feature to split the data set
    '''
    def choose_best_feature_to_split(self, data, flag):
        feature_num = len(data[0])-1
        best_gini = 10000000.0
        best_feature_index = -1
        for i in range(feature_num):
            features = [vec[i] for vec in data]
            unique_features = set(features)
            gini = 0.0
            for feature_val in unique_features:
                sub = self.split_data_set(data, i, feature_val)
                prob = len(sub)/float(len(data))
                gini_prob = len(self.split_data_set(sub, -1, flag))/float(len(sub))
                gini += prob * 2 * gini_prob * (1 - gini_prob)
            if gini < best_gini:
                best_feature_index = i
                best_gini = gini
        return best_feature_index

    '''
    :param class_list
        vote with the help of class_list
    :return
        category
    '''
    def majority_cnt(self, class_list):
        class_count = {}
        for vote in class_list:
            if vote not in class_count.keys():
                class_count[vote] = 0
            else:
                class_count[vote] += 1
        ret = sorted(class_count.iteritems(), key=itemgetter(1), reverse=True)
        return ret[0][0]

    '''
    :param data
        data set
    :param labels
        feature name list
    :param flag
        category which is used for gini
    :return
        classification and regression tree model
    '''
    def train(self, data, labels, flag):
        class_list = [vec[-1] for vec in data]
        if class_list.count(class_list[0]) == len(class_list):
            return class_list[0]
        if len(data[0]) == 1:
            return self.majority_cnt(class_list)
        best_feature_index = self.choose_best_feature_to_split(data, flag)
        best_feature_label = labels[best_feature_index]
        tree = {best_feature_label: {}}
        sub_labels = labels[:best_feature_index]
        sub_labels.extend(labels[best_feature_index+1:])
        features = [vec[best_feature_index] for vec in data]
        unique_vals = set(features)
        for value in unique_vals:
            tree[best_feature_label][value] = \
                self.train(self.split_data_set(data, best_feature_index, value), sub_labels, flag)
        return tree

    '''
    :param model
        classification and regression tree
    :param labels
        feature labels
    :param test_set
        one test case
    :param flag
        category which is used for gini
    :param
        category
    '''
    def classify(self, model, labels, test_set, flag):
        first_str = list(model.keys())[0]
        second_dict = model[first_str]
        feature_index = labels.index(first_str)
        class_label = flag
        for key in second_dict.keys():
            if test_set[feature_index] == key:
                if type(second_dict[key]).__name__ == 'dict':
                    class_label = self.classify(second_dict[key], labels, test_set, flag)
                else:
                    class_label = second_dict[key]
        return class_label

    '''
    :param model
        classification and regression tree model
    :param labels
        feature labels
    :param test_data
        data set for testing
    :param flag
        category which is used for gini
    :return
        category
    '''
    def classify_all(self, model, labels, test_data, flag):
        class_label_all = []
        for test_set in test_data:
            class_label_all.append(self.classify(model, labels, test_set, flag))
        return class_label_all

if __name__ == '__main__':
    dataSet = [[0, 0, 0, 0, 'N'],
               [0, 0, 0, 1, 'N'],
               [1, 0, 0, 0, 'Y'],
               [2, 1, 0, 0, 'Y'],
               [2, 2, 1, 0, 'Y'],
               [2, 2, 1, 1, 'N'],
               [1, 2, 1, 1, 'Y']]
    labels = ['outlook', 'temperature', 'humidity', 'windy']
    testSet = [[0, 1, 0, 0],
               [0, 2, 1, 0],
               [2, 1, 1, 0],
               [0, 1, 1, 1],
               [1, 1, 0, 1],
               [1, 0, 1, 0],
               [2, 1, 0, 1]]
    model = CARTDTree()

    tree = model.train(dataSet, labels, 'N')
    print tree
    print model.classify_all(tree, labels, testSet, 'N')

