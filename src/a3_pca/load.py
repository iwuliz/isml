# Author: Wuli Zuo, a1785343
# Date: 2020-09-24


import numpy as np


# load data set from a data file
def load_data_set(filepath):
    data_set = np.loadtxt(filepath, dtype=float, delimiter=',')
    label_arr, data_arr = np.split(data_set, indices_or_sections=(1,), axis=1)
    return data_arr, label_arr
