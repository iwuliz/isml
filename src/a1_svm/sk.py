# Author: Wuli Zuo, a1785343
# Date: 2020-08-18 16:36:02

import numpy as np
from cvxopt import solvers
from cvxopt import matrix as mx
from sklearn import svm


# function of training svm with sklearn
def svm_sk(data_train, label_train, regularisation_para_C):
    # train svm classifier
    # kernel='linear'，ovo: one vs. one
    sk_svm_model = svm.SVC(C=regularisation_para_C, kernel='linear', decision_function_shape='ovo')
    sk_svm_model.fit(data_train, label_train.ravel())

    # decision function
    decision_f_param = sk_svm_model.decision_function(data_train)

    # save svm as a file for analysis
    np.savetxt('../../output/sk_svm_model', decision_f_param, fmt="%.8f", delimiter=',')

    print('Train decision function:\n', decision_f_param)

    return sk_svm_model
