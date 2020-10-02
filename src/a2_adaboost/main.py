# Author: Wuli Zuo, a1785343
# Date: 2020-09-11

import numpy as np
import plot
import load
import adaboost
from src.a2_adaboost import predict
from src.a2_adaboost import sk
import time

# load data
data_train, label_train, data_test, label_test = load.load_data_set('../../data/wdbc_data.csv')

# train AdaBoost
print('\n# 1. My Adaboost')
print('\n## Use decision stump as weak classifiers')
start = time.time()
adaboost_model, err_train = adaboost.adaboost_train(data_train, label_train)
end = time.time()
training_time = end-start
print('\n## AdaBoost model\n', adaboost_model)
first_0_at = np.where(err_train == 0)[0][0]
print('  * First 0-err appears at iteration ', first_0_at)

# predict with AdaBoost
start = time.time()
err_test = predict.adaboost_predict(data_test, label_test, adaboost_model)
end = time.time()
test_time = end-start
print('\n## Predict error from 1-100 weak classifier(s)\n', err_test)
print('\n  * Error at Iteration %d: %s' % (first_0_at, err_test[first_0_at]))
print('  * Min error at Iteration %d: %s' % (err_test.argmin(), err_test.min()))
print('  * Avg error from Iteration 15-100: %s' % (err_test[15:].mean()))
print('\n## Running time')
print('   * Training time: %.2fs' % training_time)
print('   * Test time: %.2fs' % test_time)
# plot error rate
plot.plot_err_round(err_train, err_test, 'training', 'test', 'My AdaBost')

# train and predict with sklearn AdaBoost to compare
print('\n# 2. Sklearn AdaBoost')
sk.adaboost_sk(data_train, label_train, data_test, label_test)


# train and predict with sklearn SVM to compare
print('\n# 3. Sklearn SVM')
sk_svm = sk.svm_sk(data_train, label_train)
print('\n## Performance')
print('   * Training error：', 1-sk_svm.score(data_train, label_train))
start = time.time()
print('   * Test error：', 1-sk_svm.score(data_test, label_test))
end = time.time()
print('\n## Test time: %.2fs' % (end - start))

