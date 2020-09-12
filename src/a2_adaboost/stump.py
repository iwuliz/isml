# Author: Wuli Zuo, a1785343
# Date: 2020-09-11


import numpy as np

# classify by one-dim
def stump_classify(data, feature_id, thresh, LOR):
    predict = np.ones((data.shape[0],1))
    if LOR == 'left':
        predict[data[:,feature_id] <= thresh] = -1.0
    else:
        predict[data[:,feature_id] > thresh] = -1.0
    return predict


# build a stump from one-dim and return the best, error and predict result
def build_stump(data, label, weight):
    sample_num = data.shape[0]
    dim_num = data.shape[1]
    step_num = 20  # tried from [5, 10, 15, 20, 30, 40, 50]
    err_min = float('inf')
    best_stump = {}
    best_label = np.zeros(sample_num)

    for j in range(dim_num):
        min_j, max_j = data[:,j].min(), data[:,j].max()
        step = (max_j-min_j)/step_num
        for i in range(-1, step_num+1):
            thresh = min_j+i*step
            for LOR in ['left', 'right']:
                predict = stump_classify(data, j, thresh, LOR)
                errs = np.ones((sample_num,1))
                errs[predict == label] = 0
                err_value = np.dot(weight.T, errs)[0]
                if err_value < err_min:
                    err_min = err_value
                    best_stump['dim'] = j
                    best_stump['thresh'] = thresh
                    best_stump['LOR'] = LOR
                    best_label = predict

    return best_stump, err_min, best_label
