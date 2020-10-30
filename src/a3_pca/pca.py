# Author: Wuli Zuo, a1785343
# Date: 2020-10-24

import numpy as np


# compute eigenvalue and eigenvector
def eigen(data):
    mean = np.mean(data, axis=0)  # compute mean of a feature
    data = data - mean
    cov = np.cov(data.T)  # compute the covariance matrix
    eigen_val, eigen_vec = np.linalg.eig(cov)  # find the eigenvalue and the eigenvector
    eigen_val = np.real(eigen_val)
    eigen_vec = np.real(eigen_vec)
    index = np.argsort(-eigen_val)  # sort by eigenvalue
    return eigen_val, eigen_vec, index


# apply PCA to given data set by given number of dimensions
def pca(data, eigen_vec, index, dim):
    n = eigen_vec.shape[0]
    if dim > n:
        print("dim must lower than the feature number")
        return
    else:
        select_vec = eigen_vec[:, index[:dim]]
        select_data = np.dot(data, select_vec)
    return select_data
