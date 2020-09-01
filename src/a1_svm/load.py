# Author: Wuli Zuo, a1785343
# Date: 2020-08-18 16:17:22

# function of loading data set from a file
import numpy as np


def label(s):
    it = {b'0.00': -1, b'1.00': 1}
    return it[s]


def load_data_set(filepath):
    data_set = np.loadtxt(filepath, dtype=float, delimiter=',', converters={0: label})
    np.random.shuffle(data_set)
    label_arr, data_arr = np.split(data_set, indices_or_sections=(1,), axis=1)
    return data_arr, label_arr
