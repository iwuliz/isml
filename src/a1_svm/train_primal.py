# Author: Wuli Zuo, a1785343
# Date: 2020-08-18 16:36:02

import numpy as np
from cvxopt import solvers
from cvxopt import matrix as mx
from sklearn import svm


# function of training svm in the primal with cvxopt
def svm_train_primal(data_train, label_train, regularisation_para_C):
    m = len(data_train[0])  # features
    n = len(label_train)  # examples

    # Construct matrix/vector parameters for cvxopt

    # construct P
    p1 = 2 * np.eye(m, m)  # n阶方阵，对角线元素为2
    p2 = np.zeros((n + 1, m))  # 新建一个(n+1)*m的矩阵，以0填充
    p3 = np.zeros((m + n + 1, n + 1))
    p_tmp = np.vstack((p1, p2))  # 纵向拼接p1, p2
    p = np.hstack((p_tmp, p3))  # 横向拼接p_tmp, p3
    P = mx(p)

    # construct q
    q1 = np.zeros((m + 1, 1))
    q2 = regularisation_para_C / n * np.ones((n, 1))
    q = np.vstack((q1, q2))
    q = mx(q)

    # construct G
    g_tmp = np.ones((n, 1))
    g1 = np.hstack((data_train, g_tmp))
    for i in range(n):
        for j in range(m + 1):
            g1[i][j] = -1 * label_train[i] * g1[i][j]
    g2 = -1 * np.eye(n, n)
    g3 = np.zeros((n, m + 1))
    g_tmp1 = np.hstack((g1, g2))
    g_tmp2 = np.hstack((g3, g2))
    g = np.vstack((g_tmp1, g_tmp2))
    G = mx(g)

    # construct h
    h1 = -1 * np.ones((n, 1))
    h2 = np.zeros((n, 1))
    h = mx(np.vstack((h1, h2)))

    # solve
    sol = solvers.qp(P, q, G, h)
    classifier = sol['x'][:m + 1]

    # save svm as a file for analysis
    np.savetxt('../../output/svm_model_primal', classifier, fmt="%.8f", delimiter=',')

    # print output
    print(np.array(sol['status']))
    print(classifier)

    return classifier