__author__ = 'jiangdon'

from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import LabelBinarizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz
from com.edu.bupt.utils.FileUtils import *

class DecisionTree:

    def __init__(self):
        pass

    def __init__(self, filename):
        fileUtils = FileUtils()
        reader = fileUtils.getCsvReader(filename)
        self.headers = reader.next()
        self.label_list = []
        self.feature_list = []
        for row in reader:
            self.label_list.append(row[len(row)-1])
            rowDict = {}
            for i in range(1, len(row)-1):
                rowDict[self.headers[i]] = row[i]
            self.feature_list.append(rowDict)

    # the sklearn utils will convert the discrete feature to vector which is represent
    # as binary (like 01001000)
    def data2Vector(self):
        vec = DictVectorizer()
        dummy_x = vec.fit_transform(self.feature_list).toarray()
        lb = LabelBinarizer()
        dummy_y = lb.fit_transform(self.label_list)
        return dummy_x, dummy_y

    # here the decision tree use the algorithm which we call ID3, ID3 will use
    # information gain as feature select
    def DecisionTreeModel(self, dummy_x, dummy_y):
        clf = DecisionTreeClassifier(criterion='entropy')
        clf.fit(dummy_x, dummy_y)
        return clf


# with open('dt_information_gain.dot', 'w') as f:
#     f = export_graphviz(clf, feature_names=vec.get_feature_names(), out_file=f)




