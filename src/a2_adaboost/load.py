# Author: Wuli Zuo, a1785343
# Date: 2020-09-11


import numpy as np


# define label as {-1, +1}
def label(s):
    it = {b'M': -1, b'B': 1}
    return it[s]


# load data set from a data file
def load_data_set(filepath):
    data_set = np.loadtxt(filepath, dtype=float, delimiter=',', converters={1: label})[:, 1:]
    label_arr, data_arr = np.split(data_set, indices_or_sections=(1,), axis=1)
    data_train = np.array(data_arr[:300:, :])
    label_train = np.array(label_arr[:300, :])
    data_test = np.array(data_arr[301:, :])
    label_test = np.array(label_arr[301:, :])
    return data_train, label_train, data_test, label_test
