# Author: Wuli Zuo, a1785343
# Date: 2020-10-26


import numpy as np
import pandas as pd
import random


# compute Euclidean distance
def generate_dis(data_set, centres, k):
    dis_list = []
    for data in data_set:
        # compute distance to each centroid
        dis_arr = np.tile(data, (k, 1)) - centres
        dis_sqsum = np.sum(dis_arr * dis_arr, axis=1)
        dis = np.sqrt(dis_sqsum)
        dis_list.append(dis)
    # return the distance of each sample to each centroid as an (n*k) array
    dis_list = np.array(dis_list)
    return dis_list


# classify and compute centres
def classify(data_set, centres, k):
    # compute distances
    dis_list = generate_dis(data_set, centres, k)
    # classify and compute new centres
    min_dis_idx = np.argmin(dis_list, axis=1)
    loss = np.min(dis_list, 1).sum()
    new_centres = pd.DataFrame(data_set).groupby(min_dis_idx).mean()
    new_centres = new_centres.values
    # compute change
    change = new_centres - centres
    return change, new_centres, loss


# perform K-means
def kmeans(data_set, centres, k):
    # repeat to generate new centres until no more changes
    change, new_centres, loss = classify(data_set, centres, k)
    loss_list = [loss]
    while np.any(change != 0):
        change, new_centres, loss = classify(data_set, new_centres, k)
        loss_list.append(loss)
    centres = sorted(new_centres.tolist())

    # generate the final cluster
    clusters = []
    clusters_idx = []
    dis_list = generate_dis(data_set, centres, k)
    min_dist_idx = np.argmin(dis_list, axis=1)
    for i in range(k):
        clusters.append([])
        clusters_idx.append([])
    for i, j in enumerate(min_dist_idx):
        clusters[j].append(data_set[i])
        clusters_idx[j].append(i)
    return centres, clusters, clusters_idx, loss_list


# generate random initial centres, one from each class
def random_initial(data, n):
    idx_list = []
    labels = []
    while len(labels) != n:
        idx = random.randint(0, data.shape[0])
        label = data[idx][0]
        if label not in labels:
            labels.append(label)
            idx_list.append(idx)
    return idx_list, labels


# calculate average percentage of clusters
# measured by the percentage of samples share the same label with the initial centre of each class
def cal_pct(init_idx_list, clusters_idx, label):
    list = init_idx_list.copy()
    pcts = 0
    for i in range(10):
        for idx in list:
            if idx in clusters_idx[i]:
                labels_i = label[clusters_idx[i]]
                label_init_i = label[idx]
                init_label_tile = label_init_i * np.ones((len(clusters_idx[i]), 1))
                pct = np.array(labels_i == init_label_tile).mean()
                pcts += pct
                list.remove(idx)
                break
    pct_avg = pcts / 10
    return pct_avg

