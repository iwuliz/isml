# Author: Wuli Zuo, a1785343
# Date: 2020-08-18 16:36:02

import numpy as np
from cvxopt import solvers
from cvxopt import matrix as mx


# function of training svm in the primal with cvxopt
def svm_train_primal(data_train, label_train, regularisation_para_C):
    m = len(data_train[0])  # features
    n = len(label_train)  # examples

    # Construct matrix/vector coefficients for cvxopt
    print('\nConstructing coefficients matrix...')

    # construct P for 1/2||w||^2, (m+1+n, m+1+n) matrix
    p_top_left = np.eye(m)  # (m, m) matrix, wi*wj
    p_top_right = np.zeros((m, n + 1))  # (m, n+1) matrix, fill up with 0
    p_top = np.hstack((p_top_left, p_top_right))
    p_bottom = np.zeros((n + 1, m + 1 + n))  # (n+1, m+1+n) matrix, fill up with 0
    P = mx(np.vstack((p_top, p_bottom)))

    # construct q for C/n*sum(i)(slack[i]), (m+1+n, 1) matrix
    q_up = np.zeros((m + 1, 1))  # (m+1, 1) matrix, w and bias
    q_down = regularisation_para_C / n * np.ones((n, 1))  # (n, 1) matrix, C/n*slacks
    q = mx(np.vstack((q_up, q_down)))

    # construct G for combine(-y[i]*x[i]-b-slack[i] < -1, -slack[i] < 0), (2n, m+1+n) matrix
    g_b = np.ones((n, 1))
    g_top_left = - label_train * np.hstack((data_train, g_b))  # -y*x-b
    g_down_left = np.zeros((n, m + 1))
    g_half_right = - np.eye(n)  # -slack
    g_top = np.hstack((g_top_left, g_half_right))  # -y[i]*x[i]-b-slack[i] < -1
    g_bottom = np.hstack((g_down_left, g_half_right))  # -slack[i] < 0
    G = mx(np.vstack((g_top, g_bottom)))

    # construct h, for combine(-y[i]*x[i]-b-slack[i] < -1, -slack[i] < 0), (2n, 1) matrix
    h_top = - np.ones((n, 1))  # -y*x-b-slack < -1
    h_bottom = np.zeros((n, 1))  # -slack < 0
    h = mx(np.vstack((h_top, h_bottom)))

    # solve
    print('Solving primal optimal...')
    sol = solvers.qp(P, q, G, h)
    print(np.array(sol['status']))
    classifier = np.array(sol['x'][:m + 1])
    print(classifier)

    # save svm as a file for analysis
    np.savetxt('../../output/svm_model_primal', classifier[::, :1], fmt="%.16f", delimiter=',')

    return classifier
