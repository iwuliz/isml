# Author: Wuli Zuo, a1785343
# Date: 2020-08-18 16:36:02

import numpy as np
from sklearn import svm


# function of training svm with sklearn
def svm_sk(data_train, label_train, regularisation_para_C):

    # train svm classifier
    # kernel='linear'，ovo: one vs. one
    svm_model = svm.SVC(C=regularisation_para_C, kernel='linear', decision_function_shape='ovo')
    svm_model.fit(data_train, label_train.ravel())

    # decision function
    # decision_f_param = sk_svm_model.decision_function(data_train)
    # print('\nDecision function:\n', decision_f_param)

    classifier = np.vstack((svm_model.coef_.T, svm_model.intercept_))
    print('\nw, b: ')
    print(classifier[::, :1])

    # save svm as a file for analysis
    np.savetxt('../../output/svm_model_sk', classifier[::, :1], fmt="%.16f", delimiter=',')

    return svm_model, classifier
