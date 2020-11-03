# Author: Wuli Zuo, a1785343
# Date: 2020-10-24

import numpy as np


# calculate eigenvalue and eigenvector
def eigen(data):
    mean = np.mean(data, axis=0)  # subtract mean of a feature
    data = data - mean
    cov = (1/(data.shape[0]-1))*np.dot(data.T, data) # calculate the covariance matrix
    # cov1 = np.cov(data.T)
    # assert np.allclose(cov, cov1)
    eigen_val, eigen_vec = np.linalg.eig(cov)  # find the eigenvalue and the eigenvector
    eigen_val = np.real(eigen_val)
    eigen_vec = np.real(eigen_vec)
    index = np.argsort(-eigen_val)  # sort by eigenvalue
    eigen_vec = eigen_vec[:, index]
    return eigen_val, eigen_vec


def pca_reconstruct(data, eigen_vec):
    P = np.dot(data, eigen_vec)
    return P


# apply PCA to given data set by given number of dimensions
def pca_apply(data, dim):
    n = data.shape[1]
    if dim > n:
        print("dim must lower than the feature number")
        return
    else:
        select_data = data[:, :dim]
    return select_data


