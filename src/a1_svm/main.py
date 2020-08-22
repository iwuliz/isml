# Author: Wuli Zuo, a1785343
# Date: 2020-08-18 17:11:39

import load
import train_primal
import train_dual
import sk
import predict


# 1. Preparation
# load and process training data
data_train, label_train = load.load_data_set('../../data/a1_svm/tinytrain.csv')
'''
data_train = data_process.standardize(data_train)
data_train = data_process.normalization(data_train)
'''

# load and process testing data
data_test, label_test = load.load_data_set('../../data/a1_svm/test.csv')
'''
data_test = data_process.standardize(data_test)
data_test = data_process.normalization(data_test)
'''

# define regularisation parameter for slack
# default as 1; larger C gives better classification，but may overfit
regularisation_para_C = 1

# 2. Primal problem
# train svm in the primal
print("# 1. Train primal...\n")
svm_model = train_primal.svm_train_primal(data_train, label_train, regularisation_para_C)

# test svm_model on training set and testing set and print results
train_accuracy = predict.svm_predict_primal(data_train, label_train, svm_model)
test_accuracy = predict.svm_predict_primal(data_test, label_test, svm_model)
print("Training set accuracy with cvxopt:", train_accuracy)
print("Testing set accuracy with cvxopt:", test_accuracy)

# 3. Dual problem
# train svm in the dual
print("\n\n# 2. Train dual...\n")
svm_model_d = train_dual.svm_train_dual(data_train, label_train, regularisation_para_C)

# test svm_model_d on training set and testing set and print results
train_accuracy_d = predict.svm_predict_primal(data_train, label_train, svm_model_d)
test_accuracy_d = predict.svm_predict_dual(data_test, label_test, svm_model_d)
print("Training set accuracy(dual) with cvxopt:", train_accuracy_d)
print("Testing set accuracy(dual) with cvxopt:", test_accuracy_d)

# 4. Compare with sklearn
# train & test with sklearn
print("\n\n# 3. Use sklearn to train and test, compare the results:\n")
sk_svm_model = sk.svm_sk(data_train, label_train, regularisation_para_C)
# calculate accuracy
print("Training set accuracy with sklearn：", sk_svm_model.score(data_train, label_train))
print("Testing set accuracy with sklearn：", sk_svm_model.score(data_test, label_test))