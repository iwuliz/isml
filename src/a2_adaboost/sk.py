# Author: Wuli Zuo, a1785343
# Date: 2020-09-12


import numpy as np
from sklearn import svm
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
import time
import plot
import matplotlib.pyplot as plt


# sklearn grid_search and train, return sk AdaBoost model
def adaboost_sk(data_train, label_train, data_test, label_test):

    # train and predict with default parameter
    start = time.time()
    sk_boost_model = AdaBoostClassifier(DecisionTreeClassifier(max_depth=1),
                                        n_estimators=100)
    sk_boost_model.fit(data_train, label_train.ravel())
    end = time.time()
    training_time = end-start

    err_train = np.ones((1, 100))
    err_train = err_train - np.array(list(sk_boost_model.staged_score(data_train, label_train)))

    start = time.time()
    err_test = np.ones((1, 100))
    err_test = err_test - np.array(list(sk_boost_model.staged_score(data_test, label_test)))
    end = time.time()
    test_time = end-start

    print('\n## Predict error with default parameter from 1-100 n_estimator(s)\n', err_test[0])
    print('\n  * Final test error：', 1 - sk_boost_model.score(data_test, label_test))
    print('  * Min error at Iteration %d: %s' % (err_test.argmin(), err_test.min()))
    print('  * Avg error from Iteration 15-100: %s' % (err_test[::, 15:].mean()))
    # plot error rate
    plot.plot_err_round(err_train[0], err_test[0], 'training', 'test', 'Sklearn Adaboost-default parameter')
    # print running time
    print('\n## Running time (with default parameter)')
    print('\n   * Training time: %.2fs' % training_time)
    print('   * Test time: %.2fs' % test_time)

    # grid search for best parameter
    print('\n## Grid search: K-fold cross validation')
    # define parameter grid
    alg = ['SAMME', 'SAMME.R']
    n = [1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 200]
    rate = [0.4, 0.6, 0.8, 1, 1.2, 1.4]
    tuned_parameters = [{'algorithm': alg, 'n_estimators': n, 'learning_rate': rate}]

    # use GridSearchCV，to search for best parameters for SVC()
    grid_search = GridSearchCV(AdaBoostClassifier(), tuned_parameters, cv=5)
    grid_search.fit(data_train, label_train.ravel())
    '''
    scores = grid_search.cv_results_['mean_test_score']
    params = grid_search.cv_results_['params']
    for score, param in zip(scores, params):
        print('   Parameters: ', param)
        print('       mean score: %.2f%%' % float(100 * score))
    '''
    # print best parameter and best score
    print('   * Best parameters:', grid_search.best_params_)
    print('   * Best score: %.2f%%' % (100 * grid_search.best_score_))

    # train and predict with best parameter
    start = time.time()
    sk_boost_model = AdaBoostClassifier(DecisionTreeClassifier(max_depth=1),
                                        algorithm=grid_search.best_params_['algorithm'],
                                        n_estimators=grid_search.best_params_['n_estimators'],
                                        learning_rate=grid_search.best_params_['learning_rate'])
    sk_boost_model.fit(data_train, label_train.ravel())
    end = time.time()
    training_time = end-start

    err_train = np.ones((1, grid_search.best_params_['n_estimators']))
    err_train = err_train - np.array(list(sk_boost_model.staged_score(data_train, label_train)))

    start = time.time()
    err_test = np.ones((1, grid_search.best_params_['n_estimators']))
    err_test = err_test - np.array(list(sk_boost_model.staged_score(data_test, label_test)))
    end = time.time()
    test_time = end-start

    print('\n## Predict error with best parameter from 1-100 n_estimator(s)\n', err_test[0])
    print('\n  * Final test error：', 1 - sk_boost_model.score(data_test, label_test))
    print('  * Min error at Iteration %d: %s' % (err_test.argmin(), err_test.min()))
    print('  * Avg error from Iteration 15-100: %s' % (err_test[::, 15:].mean()))
    # plot error rate
    plot.plot_err_round(err_train[0], err_test[0], 'training', 'test', 'Sklearn Adaboost-best parameter')
    print('\n## Running time (with best parameter)')
    # print running time
    print('\n   * Training time: %.2fs' % training_time)
    print('   * Test time: %.2fs' % test_time)

    # analyse parameters' effects on performance
    fig = plt.figure(figsize=(15, 10))
    fig.suptitle('Algorithm & learning_rate - error')
    for i, learning_rate in enumerate([0.6, 0.8, 1, 1.2]):
        ax = fig.add_subplot(2, 2, i+1)
        for algorithm in alg:
            clf = AdaBoostClassifier(DecisionTreeClassifier(max_depth=1),
                                     algorithm=algorithm,
                                     n_estimators=100,
                                     learning_rate=learning_rate)
            clf.fit(data_train, label_train.ravel())
            err_train = np.ones((1, grid_search.best_params_['n_estimators']))
            err_test = np.ones((1, grid_search.best_params_['n_estimators']))
            err_train = err_train - np.array(list(clf.staged_score(data_train, label_train)))
            err_test = err_test - np.array(list(clf.staged_score(data_test, label_test)))
            # plot
            x = range(1, grid_search.best_params_['n_estimators']+1)
            ax.plot(x, err_train[0], label="%s training error" % algorithm)
            ax.plot(x, err_test[0], label="%s testing error" % algorithm)
        ax.set_title("learning_rate: %.2f" % learning_rate)
        ax.set_xlabel("estimators")
        ax.set_ylabel("error rate")
        plt.legend(loc="best")
    plt.savefig('../../output/Algorithm&learning_rate-error.png')
    plt.show()

    return sk_boost_model


# sklearn grid_search and train, return sk SVM model
def svm_sk(data_train, label_train):
    print('\n## Grid search: K-fold cross validation')
    c = [0.001, 0.01, 0.1, 1, 10, 100]
    gamma = [0.001, 0.01, 0.1, 1, 10, 100]

    tuned_parameters = [{'kernel': ['rbf'], 'gamma': gamma, 'C': c},
                        {'kernel': ['linear'], 'C': c}]

    # use GridSearchCV，to search for best parameters for SVC()
    grid_search = GridSearchCV(svm.SVC(), tuned_parameters, cv=5)
    grid_search.fit(data_train, label_train.ravel())

    # print best parameter and best score
    print('   * Best parameters:', grid_search.best_params_)
    print('   * Best score: %.2f%%' % (100 * grid_search.best_score_))

    # train svm classifier with best param
    start = time.time()
    if grid_search.best_params_['kernel'] == 'linear':
        sk_svm_model = svm.SVC(C=grid_search.best_params_['C'], kernel=grid_search.best_params_['kernel'])
    else:
        sk_svm_model = svm.SVC(C=grid_search.best_params_['C'], kernel=grid_search.best_params_['kernel'],
                               gamma=grid_search.best_params_['gamma'])
    sk_svm_model.fit(data_train, label_train.ravel())
    end = time.time()
    print('\n## Training time: %.2fs' % (end - start))

    return sk_svm_model
