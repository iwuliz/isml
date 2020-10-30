# Author: Wuli Zuo, a1785343
# Date: 2020-10-28


import numpy as np


# add given columns of gaussian noise to original data
def add_gauss(data, add):
    m = data.shape[0]
    # use mean and standard deviation to generate gaussian noise
    noise = data.std()*np.random.randn(m, add)+data.mean()
    # append noise to original data
    data_polluted = np.hstack((data, noise))
    return data_polluted
