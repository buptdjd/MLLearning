import numpy as np
import cPickle as pickle


# This is enhance cluster algorithm which we call k-means++
# k-means++ is an algorithm for choosing the initial values (or "seeds")
# for the k-means clustering algorithm.
class ExtKmeans:
    def __init__(self):
        pass

    # random select initial seeds
    def random_select(self, dis):
        sum = 0
        for i, p in enumerate(dis):
            sum += dis[i]
        sum *= np.random.random()
        for i, p in enumerate(dis):
            sum -= p
            if sum > 0:
                continue
            return i
        return 0

    # @param data: point
    # @param cluster_centers: cluster center points
    # @param k: amount of cluster centers
    # @return: find the closest cluster center for point
    def nearest(self, data, cluster_centers, k):
        idx = 0
        d_min = np.sum((data-cluster_centers[0])**2)
        for i in range(1, k):
            d = np.sum((data-cluster_centers[i])**2)
            if d_min > d:
                idx = i
                d_min = d
        return idx

    # @param data: raw data which expect to be clustered
    # @param k: amount of cluster centers
    # @param iteration: iteration of training
    # @return: cluster labels for data
    def train(self, data, k, iteration):
        m = len(data)
        n = len(data[0])
        cluster_centers = np.zeros((k, n))
        j = np.random.randint(m)
        cluster_centers[0] = data[j, :]
        dis = np.zeros(m)-1
        i = 0
        while i < k-1:
            for j in range(m):
                d = np.sum((cluster_centers[i]-data[j])**2)
                if (dis[j] < 0) or (dis[j] > d):
                    dis[j] = d
            j = self.random_select(dis)
            i += 1
            cluster_centers[i] = data[j, :]
        clusters = np.zeros(m, dtype=np.int)-1
        cc = np.zeros((k, n))
        c_number = np.zeros(k)
        for times in range(iteration):
            for i in range(m):
                c = self.nearest(data[i], cluster_centers, k)
                clusters[i] = c
                c_number[c] += 1
                cc[c] += data[i]
            for i in range(k):
                cluster_centers[i] = cc[i]/c_number[i]
            cc.flat = 0
            c_number.flat = 0
        return clusters

if __name__ == '__main__':
    path = r"D:/github/MLLearning/datasets/cluster.pkl"
    with open(path) as inf:
        samples = pickle.load(inf)
    N = 0
    for smp in samples:
        N += len(smp[0])
    X = np.zeros((N, 2))
    idxfrm = 0
    for i in range(len(samples)):
        idxto = idxfrm + len(samples[i][0])
        X[idxfrm:idxto, 0] = samples[i][0]
        X[idxfrm:idxto, 1] = samples[i][1]
        idxfrm = idxto
    kmeansplusplus = ExtKmeans()
    k = 3
    iteration = 300
    clusters = kmeansplusplus.train(X, k, iteration)
