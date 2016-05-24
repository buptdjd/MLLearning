from math import log

class ID3DTree:
    def __init__(self):
        pass

    def __init__(self, data):
        self.data = data

    def cal_shannon_entropy(self, data):
        numbers = len(data)
        labelCount = {}
        for vec in data:
            cur_label = vec[-1]
            if cur_label not in labelCount.keys():
                labelCount[cur_label] = 1
            else:
                labelCount[cur_label] += 1
        shannon_entropy = 0.0
        for key in labelCount:
            prob = float(labelCount[key])/numbers
            shannon_entropy -= prob * log(prob, 2)
        return prob

    def split_datasets(self, dataset, axis, value):
        ret = []
        for vec in dataset:
            if vec[axis] == value:
                reducedVec = vec[:axis]
                reducedVec.extend(vec[axis+1:])
                ret.append(reducedVec)
        return ret

    def choose_best_feature_split(self, dataset):
        feature_num = len(dataset[0])-1
        base_entropy = self.cal_shannon_entropy()
        best_info_gain = 0
        best_feature = -1
        for i in range(feature_num):
            feature_set = set([example[i] for example in dataset])
            new_entropy = 0.0
            for value in feature_set:
                sub_dataset = self.split_datasets(dataset, i, value)
                prob = len(sub_dataset)/float(len(dataset))
                new_entropy += prob * self.cal_shannon_entropy(sub_dataset)
            info_gain = base_entropy - new_entropy
            if info_gain > best_info_gain:
                best_info_gain = info_gain
                best_feature = i
        return best_feature
