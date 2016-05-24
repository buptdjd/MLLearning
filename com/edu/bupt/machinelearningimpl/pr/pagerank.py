
class Node:
    def __init__(self, id, pk):
        self.id = id
        self.pk = pk

class PageRank:

    def __init__(self):
        pass

    # map phrase of pagerank
    def pk_map(self, input):
        output = {}
        for node, outlinks in input.items():
            size = len(outlinks)
            for link in outlinks:
                if link in output:
                    output[link] += (float)(node.pk)/size
                else:
                    output[link] = (float)(node.pk)/size

        return output

    # reduce phrase of pagerank
    def pk_reduce(self, input):
        for result in input:
            for node, value in result.items():
                node.pk += value


    def pk_clear(self, nodes):
        for node in nodes:
            node.pk = 0

    def pk_last(self, nodes):
        last_nodes = []
        for node in nodes:
            last_nodes.append(Node(node.id, node.pk))
        return last_nodes

    def pk_diff(self, nodes, last_nodes):
        diff = 0
        for i in range(len(nodes)):
            print('node pk %f, last_node pk %f' %(nodes[i].pk, last_nodes[i].pk))
            diff += abs(nodes[i].pk - last_nodes[i].pk)
        return diff

    def pk_test1(self):
        node1 = Node(1, 0.25)
        node2 = Node(2, 0.25)
        node3 = Node(3, 0.25)
        node4 = Node(4, 0.25)
        nodes = [node1, node2, node3, node4]
        threshold = 0.0001
        max_iters = 100

        for iter_count in range(max_iters):
            iter_count += 1
            lastnodes=self.pk_last(nodes)
            print('============ map count %d =================' % (iter_count))
            in1 = {node1: [node2, node3, node4], node2: [node3, node4]}
            in2 = {node3: [node1, node4], node4: [node2]}

            mapout1 = self.pk_map(in1)
            mapout2 = self.pk_map(in2)

            for node, value in mapout1.items():
                print str(node.id) + ' ' + str(value)

            for node, value in mapout2.items():
                print str(node.id) + ' ' + str(value)

            print('============ reduce count %d =================' % (iter_count))

            reducein = [mapout1, mapout2]
            self.pk_clear(nodes)
            self.pk_reduce(reducein)

            for node in nodes:
                print str(node.id) + ' ' + str(node.pk)

            diff=self.pk_diff(nodes,lastnodes)
            if diff < threshold:
                break

if __name__ == '__main__':
    pageRank = PageRank()
    pageRank.pk_test1()