# Author: Wuli Zuo, a1785343
# Date: 2020-09-11

import numpy as np
import plot
import load
import adaboost
from src.a2_adaboost import predict
from src.a2_adaboost import sk


# load data
data_train, label_train, data_test, label_test = load.load_data_set('../../data/wdbc_data.csv')

# train Adaboost
print('\n# 1. Decision stump as weak classifiers\n')
adaboost_model, err_train = adaboost.adaboost_train(data_train, label_train)
print('\n# 2. Adaboost model:\n\n  ', adaboost_model)
first_0_at = np.where(err_train == 0)[0][0]
print('  First 0-err appears at iteration ', first_0_at)

# predict with Adaboost
print('\n# 3. Adaboost predict')
err_test = predict.adaboost_predict(data_test, label_test, adaboost_model)
print('  Predict error from 1-100 weak classifier(s)\n', err_test)
print('\n  Error at iteration %d: %s' % (first_0_at, err_test[first_0_at]))
print('  Min error at iteration %d: %s' % (err_test.argmin(), err_test.min()))

# plot error rate
plot.plot_err_round(err_train, "training")
plot.plot_err_round(err_test, "test")

# train and predict with sklearn Adaboost to compare
print('\n# 4. Sklearn Adaboost')
sk_adaboost = sk.svm_adaboost(data_train, label_train)
print("  Training error：", 1-sk_adaboost.score(data_train, label_train))
print("  Test error：", 1-sk_adaboost.score(data_test, label_test))

# train and predict with sklearn SVM to compare
print('\n# 5. Sklearn SVM')
sk_svm = sk.svm_sk(data_train, label_train)
print("  Training error：", 1-sk_svm.score(data_train, label_train))
print("  Test error：", 1-sk_svm.score(data_test, label_test))
