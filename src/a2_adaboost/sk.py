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


# sklearn train with default parameter, return sk AdaBoost model and training time
def adaboost_sk_default(data_train, label_train):
    start = time.time()
    sk_boost_model = AdaBoostClassifier(DecisionTreeClassifier(max_depth=1),
                                        n_estimators=100)
    sk_boost_model.fit(data_train, label_train.ravel())
    end = time.time()
    training_time = end-start

    return sk_boost_model, training_time


def adaboost_sk_deeper(data_train, label_train, data_test, label_test):

    plt.figure(figsize=(8, 5))
    plt.title('Accuracy of different depth of weak learns')
    plt.xlabel('# rounds')
    plt.ylabel('accuracy: %')

    for depth in range(1, 6):
        start = time.time()
        sk_boost_model = AdaBoostClassifier(DecisionTreeClassifier(max_depth=depth),
                                            n_estimators=100)
        sk_boost_model.fit(data_train, label_train.ravel())
        end = time.time()
        training_time = end-start
        acc_train = np.array(list(sk_boost_model.staged_score(data_train, label_train)))
        start = time.time()
        acc_test = np.array(list(sk_boost_model.staged_score(data_test, label_test)))
        end = time.time()
        test_time = end - start

        print('\n### depth = %d' % depth)
        print('\n   * Max accuracy at Iteration %d: %s%%' % (acc_test.argmax(), 100 * acc_test.max()))
        print('   * Final test accuracy: %s%%' % float(100 * sk_boost_model.score(data_test, label_test)))
        print('   * Training time: %.2fs' % training_time)
        print('   * Test time: %.2fs' % test_time)

        x = range(0, acc_train.shape[0])
        y1 = 100 * acc_train
        y2 = 100 * acc_test
        plt.ylim(85, 101)
        plt.plot(x, y1, label='max_depth = %d, training accuracy' % depth)
        plt.plot(x, y2, label='max_depth = %d, test accuracy' % depth)

    plt.legend(loc='best')

    plt.savefig('../../output/acc&iteration with deeper weak learns')
    plt.show()


# sklearn grid_search and train with best parameters,
# analyse parameters
# return sk AdaBoost model and training time
def adaboost_sk_best(data_train, label_train, data_test, label_test):

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

    # analyse parameters' effects on performance
    fig = plt.figure(figsize=(15, 10))
    fig.suptitle('Algorithm & learning_rate - accuracy')
    for i, learning_rate in enumerate([0.6, 0.8, 1, 1.2]):
        ax = fig.add_subplot(2, 2, i+1)
        for algorithm in alg:
            clf = AdaBoostClassifier(DecisionTreeClassifier(max_depth=1),
                                     algorithm=algorithm,
                                     n_estimators=100,
                                     learning_rate=learning_rate)
            clf.fit(data_train, label_train.ravel())
            acc_train = np.array(list(clf.staged_score(data_train, label_train)))
            acc_test = np.array(list(clf.staged_score(data_test, label_test)))
            # plot
            x = range(1, grid_search.best_params_['n_estimators']+1)
            ax.plot(x, acc_train, label="%s training accuracy" % algorithm)
            ax.plot(x, acc_test, label="%s testing accuracy" % algorithm)
        ax.set_title("learning_rate: %.2f" % learning_rate)
        ax.set_xlabel("estimators")
        ax.set_ylabel("accuracy")
        plt.legend(loc="best")
    plt.savefig('../../output/Algorithm&learning_rate-accuracy.png')
    plt.show()

    return sk_boost_model, training_time


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
    training_time = end-start

    return sk_svm_model, training_time
