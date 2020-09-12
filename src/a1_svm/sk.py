# Author: Wuli Zuo, a1785343
# Date: 2020-08-18 16:36:02

import numpy as np
from sklearn import svm
from sklearn.model_selection import GridSearchCV
import draw


# function of training svm with sklearn
def svm_sk(data_train, label_train, mode):
    if mode == '0':
        print('\n## Grid search: decide optimal C for sklearn SVM')
        Cs = [0.0001, round(1/8500,8), 0.0005, round(5/8500,8), 0.001, 0.01, 0.1, 1]
        tuned_parameters = [{'C': Cs}]

        # use GridSearchCV，to search for best parameters for SVC()
        grid_search = GridSearchCV(svm.SVC(kernel='linear', decision_function_shape='ovo'), tuned_parameters, cv=5)
        # train
        grid_search.fit(data_train, label_train.ravel())
        scores = grid_search.cv_results_['mean_test_score']
        params = grid_search.cv_results_['params']
        for score, param in zip(scores, params):
            print('## K-fold cross validation，', param)
            print('     mean score: %.2f%%' % float(100 * score))
        draw.draw_C_acc(Cs, scores, "sk")

        # print best parameter and best score
        best_C = grid_search.best_params_['C']
        print('\n* Best parameters:', grid_search.best_params_)
        print('* Best score: %.2f%%' % (100 * grid_search.best_score_))
    else:
        best_C = round(5/8500, 8)

    # train svm classifier with best C
    # kernel='linear'，ovo: one vs. one
    svm_model = svm.SVC(C=best_C, kernel='linear', decision_function_shape='ovo')
    svm_model.fit(data_train, label_train.ravel())

    # decision function
    # decision_f_param = sk_svm_model.decision_function(data_train)
    # print('\nDecision function:\n', decision_f_param)

    classifier = np.vstack((svm_model.coef_.T, svm_model.intercept_))
    if mode != '2':
        print('\n* w, b: ')
        print(classifier[::, :1])

    # save svm as a file for analysis
    np.savetxt('../../output/svm_model_sk', classifier[::, :1], fmt="%.16f", delimiter=',')

    return svm_model, classifier
