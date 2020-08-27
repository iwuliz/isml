# Author: Wuli Zuo, a1785343
# Date: 2020-08-19 11:05:08

import numpy as np


def svm_model_predict(data, label, model):
    m = len(data[0])  # features
    n = len(label)  # samples / labels
    w = model[:m]
    b = model[m]

    predict = np.sign(np.dot(data, w) + b)
    mistake = np.sum(predict != label)
    print('     correct: %d | mistake: %d | total: %d ' % (n-mistake, int(mistake), n))
    accuracy = 1 - mistake / n
    # print(predict)  # output predict results

    return accuracy


# function of testing with svm_model
def svm_predict_primal(data_test, label_test, svm_model):
    return svm_model_predict(data_test, label_test, svm_model)


# function of testing with svm_model_d
def svm_predict_dual(data_test, label_test, svm_model_d):
    return svm_model_predict(data_test, label_test, svm_model_d)
