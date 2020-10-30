# Author: Wuli Zuo, a1785343
# Date: 2020-10-26


from sklearn.decomposition import PCA
from sklearn import neighbors
from sklearn import svm
from sklearn.model_selection import GridSearchCV


def sk_pca(data_train, data_test, k):
    pca = PCA(n_components=k)
    select_data_train = pca.fit_transform(data_train)
    # print(pca.explained_variance_ratio_)
    select_data_test = pca.transform(data_test)
    return select_data_train, select_data_test


def sk_knn(data_train, label_train, data_test, label_test, k):
    # train KNN classifier
    clf = neighbors.KNeighborsClassifier(n_neighbors=1, algorithm='kd_tree')
    clf.fit(data_train, label_train.ravel())
    # predict on training data
    acc_train = clf.score(data_train, label_train)
    # predict on test data
    acc_test = clf.score(data_test, label_test)
    return acc_train, acc_test


def svm_sk(data_train, label_train):

    # use grid search to do cross validation
    # this takes hours (3106s for one k)
    # I use the best parameter for submitting version and comment c_v process
    '''
    print('\n## Grid search: K-fold cross validation')
    c = [0.1, 1, 10]
    gamma = [0.1, 0.3, 1, 10]

    tuned_parameters = [{'kernel': ['rbf'], 'gamma': gamma, 'C': c},
                        {'kernel': ['linear'], 'C': c}]

    # use GridSearchCV，to search for best parameters for SVC()
    grid_search = GridSearchCV(svm.SVC(), tuned_parameters, cv=5)
    grid_search.fit(data_train, label_train.ravel())

    scores = grid_search.cv_results_['mean_test_score']
    params = grid_search.cv_results_['params']
    for score, param in zip(scores, params):
        print('## K-fold cross validation，', param)
        print('     mean score: %.2f%%' % float(100 * score))

    # print best parameter and best score
    print('   * Best parameters:', grid_search.best_params_)
    print('   * Best score: %.2f%%' % (100 * grid_search.best_score_))

    # train svm classifier with best param
    if grid_search.best_params_['kernel'] == 'linear':
        sk_svm_model = svm.SVC(C=grid_search.best_params_['C'], kernel=grid_search.best_params_['kernel'])
    else:
        sk_svm_model = svm.SVC(C=grid_search.best_params_['C'], kernel=grid_search.best_params_['kernel'],
                               gamma=grid_search.best_params_['gamma'])
    '''
    sk_svm_model = svm.SVC(C=0.1, kernel='linear')
    sk_svm_model.fit(data_train, label_train.ravel())

    return sk_svm_model
