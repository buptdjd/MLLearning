
import numpy as np
from sklearn.decomposition import PCA


class PCAModel:
    def __init__(self):
        pass

    '''
        :param data raw data
        :param k get k principle components
        :return new data with dimensional reduction
    '''
    def dimension_reduction(self, data, k):
        m, n = data.shape
        # mean of each feature
        m_data = np.array([np.mean(data[:, i]) for i in range(n)])
        # normalization
        norm_data = data - m_data
        # scatter matrix
        # scatter matrix / (n-1) = covariance matrix
        # scatter matrix 's eigenvalues are the same to covariance matrix
        scatter_matrix = np.dot(np.transpose(norm_data), norm_data)
        eigenvalue, eigenvector = np.linalg.eig(scatter_matrix)

        eig_pairs = [(np.abs(eigenvalue[i]), eigenvector[:, i]) for i in range(n)]
        # sort eigenvector based on eigenvalue from highest to lowest
        eig_pairs.sort(reverse=True)
        # select top k eigenvector
        feature = np.array([eig[1] for eig in eig_pairs[:k]])
        # get new projection data
        projection_data = np.dot(norm_data, np.transpose(feature))
        return projection_data

if __name__ == '__main__':
    # found that pca model can get results the same to the sklearn pca
    X = np.array([[-1, 1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
    pca_sklearn = PCA(n_components=1)
    pca_sklearn.fit(X)
    print pca_sklearn.transform(X)
    pca_model = PCAModel()
    print pca_model.dimension_reduction(X, k=1)
