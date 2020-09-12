# Author: Wuli Zuo, a1785343
# Date: 2020-08-18 16:36:02

import numpy as np
from cvxopt import solvers
from cvxopt import matrix as mx


# function of training svm in the dual with cvxopt
def svm_train_dual(data_train, label_train, regularisation_para_C):
    m = len(data_train[0])  # features
    n = len(label_train)  # examples

    # Construct matrix/vector coefficients for cvxopt
    print('\nConstructing coefficients matrix...')

    # construct P for 1/2sum(i)sum(j)(a[i]a[j]y[i]y[j]x[i].Tx[j]), (n, n) matrix
    p = label_train.T * (label_train * np.dot(data_train, data_train.T))
    P = mx(p)

    # construct Q for -sum(i)a[i], (n, 1) matrix
    q = mx(- np.ones((n, 1)))

    # construct G for combine (a[i]<0, -a[i]<C/n), (2n, n) matrix
    g_top = - np.eye(n)
    g_bottom = np.eye(n)
    G = mx(np.vstack((g_top, g_bottom)))

    # construct h for combine (a[i]<0, -a[i]<C/n), (2n, 1) matrix
    h_top = np.zeros((n, 1))
    h_bottom = regularisation_para_C / n * np.ones((n, 1))
    h = mx(np.vstack((h_top, h_bottom)))

    # construct A for sum(i)y[i]a[i]=0, (1, n) matrix
    A = mx(label_train.T)

    # construct b for sum(i)y[i]a[i]=0, (1,1) matrix
    b = mx(np.array([0.]))

    # solve
    print('Solving dual optimal...')
    sol = solvers.qp(P, q, G, h, A, b)
    a = np.array(sol['x'])

    # print output
    print(np.array(sol['status']))
    # print(a)

    # compute w
    w = np.sum(a * label_train * data_train, axis=0)

    # find a support vector
    slack = 2e-6
    k = np.where((a < (regularisation_para_C/n - slack)) & (a > slack))[0][0]

    # compute b
    bias = 1 / label_train[k][0] - np.dot(w, data_train[k].T)

    # generate classifier
    classifier = np.vstack((np.array([w]).T, bias))
    print('\n* w, b:')
    print(classifier[::, :1])

    # save svm as a file for analysis
    np.savetxt('../../output/svm_model_dual', classifier[::, :1], fmt="%.16f", delimiter=',')

    return classifier
