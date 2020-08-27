# Author: Wuli Zuo, a1785343
# Date: 2020-08-18 17:11:39

import load
import primal
import dual
import sk
import predict
import compare

# 0. Preparation
# load training data
data_train, label_train = load.load_data_set('../../data/a1_svm/train.csv')
# load testing data
data_test, label_test = load.load_data_set('../../data/a1_svm/test.csv')

# define regularisation parameter for slack
# default as 1; larger C gives better classification，but may overfit
regularisation_para_C = 8500

# 1. Implement Primal SVM and predict
# train svm in the primal
print('\n# 1. Primal: Implement SVM...')
svm_model = primal.svm_train_primal(data_train, label_train, regularisation_para_C)

# Predict and compute accuracy of svm_model
print('## Primal: Predict with training data set:')
train_accuracy = predict.svm_predict_primal(data_train, label_train, svm_model)
print('     accuracy: %.2f%%' % (100 * train_accuracy))
print('## Primal: Predict with testing data set:')
test_accuracy = predict.svm_predict_primal(data_test, label_test, svm_model)
print('     accuracy: %.2f%%' % (100 * test_accuracy))


# 2. Implement Dual SVM and predict
# train svm in the dual
print('\n\n# 2. Dual: Implement SVM...')
svm_model_d = dual.svm_train_dual(data_train, label_train, regularisation_para_C)

# test svm_model_d on training set and testing set and print results
print('\n## Dual: Predict with training data set:')
train_accuracy_d = predict.svm_predict_primal(data_train, label_train, svm_model_d)
print('     accuracy(dual): %.2f%%' % (100 * train_accuracy_d))
print('## Dual: Predict with testing data set:')
test_accuracy_d = predict.svm_predict_dual(data_test, label_test, svm_model_d)
print('     accuracy(dual): %.2f%%' % (100 * test_accuracy_d))


# 3. Use sklearn implementations for comparison
# train & predict with sklearn
print('\n\n# 3. Use sklearn to train and test, compare the results:')
svm_model_sk, sk_classifier = sk.svm_sk(data_train, label_train, regularisation_para_C / 8500)
# calculate accuracy
print('## Sklearn: Predict with training data set:')
print('     accuracy： %.2f%%' % (100 * svm_model_sk.score(data_train, label_train)))
print('     accuracy： %.2f%%' % (100* svm_model_sk.score(data_test, label_test)))


# 4. Compare with sklearn
print('\n\n# 4. Compare results')
print("Compare primal with dual")
compare.compare_svm(svm_model, svm_model_d, "primal", "dual")

print("Compare primal with sk")
compare.compare_svm(svm_model, sk_classifier, "primal", "sklearn")
