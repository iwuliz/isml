# Author: Wuli Zuo, a1785343
# Date: 2020-09-12


from sklearn import svm
from sklearn.ensemble import AdaBoostClassifier
#from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV


# sklearn grid_search and train, return sk Adaboost model
def svm_adaboost(data_train, label_train):

    print('\n  Grid search:')
    alg = ['SAMME', 'SAMME.R']
    n = [30, 40, 50, 100, 200, 300]
    rate = [0.5, 0.7, 1, 1.1, 1.2, 1.3, 1.5]

    tuned_parameters = [{'algorithm': alg, 'n_estimators': n, 'learning_rate': rate}]

    # use GridSearchCV，to search for best parameters for SVC()
    grid_search = GridSearchCV(AdaBoostClassifier(), tuned_parameters, cv=5)
    # train
    grid_search.fit(data_train, label_train.ravel())

    # print best parameter and best score
    print('    Best parameters:', grid_search.best_params_)
    print('    Best score: %.2f%%' % (100 * grid_search.best_score_))

    sk_boost_model = AdaBoostClassifier(algorithm=grid_search.best_params_['algorithm'],
                                        n_estimators=grid_search.best_params_['n_estimators'],
                                        learning_rate=grid_search.best_params_['learning_rate'])
    sk_boost_model.fit(data_train, label_train.ravel())

    return sk_boost_model


# sklearn grid_search and train, return sk SVM model
def svm_sk(data_train, label_train):
    print('\n  Grid search:')
    c = [0.001, 0.01, 0.1, 1, 10, 100]
    gamma = [0.001, 0.01, 0.1, 1, 10, 100]

    tuned_parameters = [{'kernel': ['rbf'], 'gamma': gamma, 'C': c},
                        {'kernel': ['linear'], 'C': c}]

    # use GridSearchCV，to search for best parameters for SVC()
    grid_search = GridSearchCV(svm.SVC(), tuned_parameters, cv=5)
    # train
    grid_search.fit(data_train, label_train.ravel())

    # print best parameter and best score
    print('    Best parameters:', grid_search.best_params_)
    print('    Best score: %.2f%%' % (100 * grid_search.best_score_))

    # train svm classifier with best param
    if grid_search.best_params_['kernel'] == 'linear':
        sk_svm_model = svm.SVC(C=grid_search.best_params_['C'], kernel=grid_search.best_params_['kernel'])
    else:
        sk_svm_model = svm.SVC(C=grid_search.best_params_['C'], kernel=grid_search.best_params_['kernel'],
                           gamma=grid_search.best_params_['gamma'])
    sk_svm_model.fit(data_train, label_train.ravel())

    return sk_svm_model
