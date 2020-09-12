# Author: Wuli Zuo, a1785343
# Date: 2020-08-30 20:56:13

import os
import sys
import numpy as np
import predict
import primal
import draw


def validate_score(data, label, k, C):
    m = len(data[0])  # features
    fold_data = np.split(data, k, axis=0)
    fold_label = np.split(label, k, axis=0)
    scores = np.zeros((k, 1))
    print('## K-fold cross validation, C = ', C)
    for i in range(k):
        data_validate = fold_data[i]
        label_validate = fold_label[i]
        data_train = np.zeros((1, m))
        label_train = np.zeros((1, 1))
        for j in range(i):
            data_train = np.vstack((data_train, fold_data[j]))
            label_train = np.vstack((label_train, fold_label[j]))
        for j in range(i + 1, k):
            data_train = np.vstack((data_train, fold_data[j]))
            label_train = np.vstack((label_train, fold_label[j]))
        data_train = data_train[1:, ::]
        label_train = label_train[1:, ::]
        sys.stdout = open(os.devnull, 'w')
        svm_model = primal.svm_train_primal(data_train, label_train, C)
        scores[i] = predict.svm_predict_primal(data_validate, label_validate, svm_model)
        sys.stdout = sys.__stdout__
        # print('     %d-th fold as validate set, accuracy: %.2f%%' % (i, 100 * validate_accuracy[i]))
    score_mean = scores.mean()
    print('     mean accuracy: %.2f%%' % float(100 * score_mean))
    return score_mean


def cross_validate(data, label, k, Cs):
    scores = []
    for C in Cs:
        scores.append(validate_score(data, label, k, C))
    draw.draw_C_acc(Cs, scores, "primal")

    scores = np.array(scores)
    best_C = Cs[scores.argmax()]
    print('\n* Best C:', best_C)
    print('* Best score:%.2f%%' % (100 * scores.max()))

    return best_C

