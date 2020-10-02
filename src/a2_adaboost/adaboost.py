# Author: Wuli Zuo, a1785343
# Date: 2020-09-12


import numpy as np
import math
import stump


# train AdaBoost
def adaboost_train(data, label):

    # initialise
    m = data.shape[0]
    D = 1 / m * np.ones((m, 1))
    predict = np.zeros((m, 1))
    t = 0
    adaboost = []
    err = []

    # repeat to train
    while t < 100:
        print('   Weak classifier ', t+1)

        # build decision stump
        best_stump, err_t, best_label = stump.build_stump(data, label, D)
        print('     best stump: ', best_stump)
        print('     err_t = ', err_t[0])

        # compute alpha_t
        a = 1 / 2 * math.log((1 - err_t) / max(err_t, 1e-16))  # to avoid overflow when err==0
        best_stump['alpha'] = a
        predict += a * best_label
        errs = np.ones((m, 1))
        errs[(predict * label) > 0] = 0
        err_t = errs.sum() / m
        err.append(err_t)
        print('     alpha_t = ', a)
        print('     wighted err = ', err_t)
        '''
        if err == 0:
            break
        '''
        # update D
        for i in range(m):
            D[i] = D[i] * math.exp(-a * best_label[i] * label[i])
        D = D / D.sum()

        adaboost.append(best_stump)

        t += 1

    return adaboost, np.array(err)
