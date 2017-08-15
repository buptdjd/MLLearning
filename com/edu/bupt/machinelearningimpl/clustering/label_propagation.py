import string


class LabelPropagation:

    # @param data_path: input data of graph contains vertex, edges and weights of edge
    # the file format is like (origin point \t destination point \t weight
    def __init__(self, data_path):
        self.path = data_path
        self.vertex_dict, self.edge_dict = self.load()

    # @return vertices and edges of graph
    def load(self):
        f = open(self.path, "r")
        vertex_dict, edge_dict = {}, {}
        for line in f.readlines():
            lines = line.strip().split("\t")
            for i in xrange(2):
                if lines[i] not in vertex_dict:
                    vertex_dict[lines[i]] = string.atoi(lines[i])
                    edge_list = []
                    if len(lines) == 3:  # whether edges contain weight or not
                        edge_list.append(lines[1-i] + ":" + lines[2])
                    else:
                        edge_list.append(lines[1-i] + ":" + "1")
                    edge_dict[lines[i]] = edge_list
                else:
                    edge_list = edge_dict[lines[i]]
                    if len(lines) == 3:
                        edge_list.append(lines[i-1] + ":" + lines[2])
                    else:
                        edge_list.append(lines[i-1] + ":" + "1")

                    edge_dict[lines[i]] = edge_list
        return vertex_dict, edge_dict

    # @param adjacency_node_list: the adjacency vertex list of the specific vertex
    # @return community id of specific vertex
    def get_max_community_label(self, adjacency_node_list):
        label_dict = {}
        for node in adjacency_node_list:
            edge_info = node.strip().split(":")
            vertex = edge_info[0]
            weight = edge_info[1]
            if self.vertex_dict[vertex] not in label_dict:
                label_dict[self.vertex_dict[vertex]] = weight
            else:
                label_dict[self.vertex_dict[vertex]] += weight
        sort_list = sorted(label_dict.items(), key=lambda d: d[1], reverse=True)
        return sort_list[0][0]

    # @return : whether program to do label propagation or not
    def check(self):
        for vertex in self.vertex_dict.keys():
            adjacency_node_list = self.edge_dict[vertex]
            vertex_label = self.vertex_dict[vertex]
            label_check = {}
            for ad_node in adjacency_node_list:
                edge_info = ad_node.strip().split(":")
                _vertex = edge_info[0]
                if self.vertex_dict[_vertex] not in label_check:
                    label_check[self.vertex_dict[_vertex]] = 1
                else:
                    label_check[self.vertex_dict[_vertex]] += 1
            sort_list = sorted(label_check.items(), key=lambda k: k[1], reverse=True)
            if vertex_label == sort_list[0][0]:
                continue
            else:
                return 0
        return 1

    def run(self):
        t = 0
        while True:
            if self.check() == 0:
                t = t+1
                print "iteration: ", t
                for vertex in self.vertex_dict.keys():
                    adjacency_node_list = self.edge_dict[vertex]
                    self.vertex_dict[vertex] = self.get_max_community_label(adjacency_node_list)
            else:
                break
        return self.vertex_dict


if __name__ == "__main__":
    path = "/Users/buptdjd/PycharmProjects/MLLearning/datasets/lpa_test_data"
    lpa = LabelPropagation(path)
    vertex_dict = lpa.run()
    for key in vertex_dict.keys():
        print key, "------->", str(vertex_dict[key])
