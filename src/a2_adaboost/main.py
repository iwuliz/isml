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
print('\n# 1. My AdaBoost')
print('\n## Use decision stump as weak classifiers')
start = time.time()
adaboost_model, err_train_adaboost = adaboost.adaboost_train(data_train, label_train)
end = time.time()
training_time = end-start
print('\n## AdaBoost model\n', adaboost_model)
first_0_at = np.where(err_train_adaboost == 0)[0][0]
print('  * First 100-acc appears at iteration ', first_0_at)
acc_train_adaboost = np.ones((1, 100))
acc_train_adaboost = acc_train_adaboost - err_train_adaboost

# predict with AdaBoost
start = time.time()
err_test_adaboost = predict.adaboost_predict(data_test, label_test, adaboost_model)
end = time.time()
test_time = end-start
acc_test_adaboost = np.ones((1, 100))
acc_test_adaboost = acc_test_adaboost - err_test_adaboost
print('\n## Predict accuracy from 1-100 weak classifier(s)\n', acc_test_adaboost[0])
print('\n  * Accuracy at Iteration %d: %s%%' % (first_0_at, 100*acc_test_adaboost[0][first_0_at]))
print('  * Max accuracy at Iteration %d: %s%%' % (acc_test_adaboost[0].argmax(), 100*acc_test_adaboost[0].max()))
print('  * Avg accuracy from Iteration 15-100: %s%%' % (100*acc_test_adaboost[0][15:].mean()))
print('  * Final test accuracy: %s%%' % float(100*acc_test_adaboost[0][99]))
print('\n## Running time')
print('   * Training time: %.2fs' % training_time)
print('   * Test time: %.2fs' % test_time)
# plot accuracy
plot.plot_acc_round(acc_train_adaboost[0], acc_test_adaboost[0], 'training', 'test', 'My AdaBoost')

# train and predict with sklearn AdaBoost to compare
print('\n# 2. Sklearn AdaBoost')
# train and predict with default parameters
sk_adaboost_default, training_time = sk.adaboost_sk_default(data_train, label_train, data_test, label_test)
acc_train_sk_default = np.array(list(sk_adaboost_default.staged_score(data_train, label_train)))
start = time.time()
acc_test_sk_default = np.array(list(sk_adaboost_default.staged_score(data_test, label_test)))
end = time.time()
test_time = end - start

print('\n## Predict accuracy with default parameter from 1-100 n_estimator(s)\n', acc_test_sk_default)
print('\n  * Max accuracy at Iteration %d: %s%%' % (acc_test_sk_default.argmax(), 100*acc_test_sk_default.max()))
print('  * Avg error from Iteration 15-100: %s%%' % (100*acc_test_sk_default[15:].mean()))
print('  * Final test accuracy: %s%%' % float(100*sk_adaboost_default.score(data_test, label_test)))
print('\n## Running time (with default parameter)')
print('\n   * Training time: %.2fs' % training_time)
print('   * Test time: %.2fs' % test_time)
# plot accuracy rate
plot.plot_acc_round(acc_train_sk_default, acc_test_sk_default, 'training', 'test',
                    'Sklearn AdaBoost-default parameter')

# plot to compare my AdaBoost and sklearn Adaboost accuracy
plot.plot_acc_round_compare(acc_train_adaboost[0], acc_test_adaboost[0], acc_train_sk_default, acc_test_sk_default,
                            'training', 'test', 'My AdaBoost', 'Sklearn')

# train and predict with best parameters
sk_adaboost_best, training_time = sk.adaboost_sk_best(data_train, label_train, data_test, label_test)
acc_train_sk_best = np.array(list(sk_adaboost_best.staged_score(data_train, label_train)))
start = time.time()
acc_test_sk_best = np.array(list(sk_adaboost_best.staged_score(data_test, label_test)))
end = time.time()
test_time = end - start

print('\n## Predict accuracy with best parameter from 1-100 n_estimator(s)\n', acc_test_sk_best)
print('\n  * Max accuracy at Iteration %d: %s%%' % (acc_test_sk_best.argmax(), 100*acc_test_sk_best.max()))
print('  * Avg accuracy from Iteration 15-100: %s%%' % (100*acc_test_sk_best[15:].mean()))
print('  * Final test accuracy: %s%%' % float(100*sk_adaboost_best.score(data_test, label_test)))
print('\n## Running time (with best parameter)')
print('\n   * Training time: %.2fs' % training_time)
print('   * Test time: %.2fs' % test_time)
# plot accuracy rate
plot.plot_acc_round(acc_train_sk_best, acc_test_sk_best, 'training', 'test', 'Sklearn AdaBoost-best parameter')


# train and predict with sklearn SVM to compare
print('\n# 3. Sklearn SVM')
sk_svm, training_time = sk.svm_sk(data_train, label_train)
print('\n## Performance')
print('   * Training accuracy: %s%%' % float(100*sk_svm.score(data_train, label_train)))
start = time.time()
print('   * Test accuracy: %s%%' % float(100*sk_svm.score(data_test, label_test)))
end = time.time()
test_time = end-start
print('\n## Running time')
print('\n   * Training time: %.2fs' % training_time)
print('   * Test time: %.2fs' % test_time)
