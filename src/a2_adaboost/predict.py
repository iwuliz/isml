# Author: Wuli Zuo, a1785343
# Date: 2020-09-12


import numpy as np
import stump


# predict with AdaBoost model consist of 1-100 weak classifiers and return the errors
def adaboost_predict(data, label, model):
    m = data.shape[0]
    predict = np.zeros((m,1))
    err = []
    # predict
    for weak in model:
        errs = np.ones((m, 1))
        predict += stump.stump_classify(data, weak['dim'], weak['thresh'], weak['LOR']) * weak['alpha']
        # compute err
        errs[(predict * label) > 0] = 0
        err_t = errs.sum() / m
        err.append(err_t)
    return np.array(err)
