# Author: Wuli Zuo, a1785343
# Date: 2020-08-19 21:27:30

import numpy as np


def standardize(data):
    mean = np.mean(data, axis=0)
    sigma = np.std(data, axis=0)
    return (data - mean) / sigma


def normalization(data):
    _range = np.max(abs(data), axis=0)
    return data / _range
