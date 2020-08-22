# Author: Wuli Zuo, a1785343
# Date: 2020-08-18 16:36:02

import numpy as np
from cvxopt import solvers
from cvxopt import matrix as mx
from sklearn import svm


# function of training svm in the dual with cvxopt
def svm_train_dual(data_train, label_train, regularisation_para_C):
    m = len(data_train[0])  # features
    n = len(label_train)  # examples

    # Construct matrix/vector parameters for cvxopt
    # construct P
    print("Constructing P (size of n*n), may take long...")
    data_train = mx(data_train)
    p = data_train * data_train.trans()/2
    print("...finished 1/2 x.T * x...")
    p[::n+1] *= 2
    print("...finished diagonal process...")
    y_diag = mx(np.eye(n,n)*label_train)
    P = y_diag * p * y_diag
    print("...finished multiply yi to each line and column...\n")

    # construct Q
    q = mx(-1 * np.ones((n, 1)))

    # construct G
    g1 = -1 * np.eye(n, n)
    g2 = np.eye(n, n)
    g = np.vstack((g1, g2))
    G = mx(g)

    # construct h
    h1 = np.zeros((n, 1))
    h2 = regularisation_para_C * np.ones((n, 1))
    h = mx(np.vstack((h1, h2)))

    # construct A
    A = mx(label_train.T)

    # construct b
    b = mx(np.zeros((1, 1)))

    # solve
    sol = solvers.qp(P, q, G, h, A, b)
    a = np.array(sol['x'])

    # print output
    print(np.array(sol['status']))
    print(a)

    # compute w
    w = np.zeros((1, m))
    for i in range(n):
        w += np.array(a[i] * label_train[i][0] * data_train[i, :])

    # find a support vector
    k = 0
    for j in range(n):
        if (a[j][0] < regularisation_para_C) & (a[j][0] > 0):
            k = j
            break

    # compute b
    s = 0
    for i in range(n):
        prd = (data_train[i, :] * data_train[k, :].trans())[0]
        s += label_train[i][0] * a[i][0] * prd
    b = label_train[k][0] - s

    # generate classifier
    classifier = []
    for wi in w[0]:
        classifier.append(wi)
    classifier.append(b)

    # save svm as a file for analysis
    np.savetxt('../../output/svm_model_dual', classifier, fmt="%.8f", delimiter=',')

    print("\nw, b:")
    print(np.array(classifier))

    return classifier
