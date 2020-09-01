# Author: Wuli Zuo, a1785343
# Date: 2020-08-18 17:11:39

import sys
import load
import primal
import dual
import sk
import predict
import compare
import validate
import numpy as np


def main(argv):
    if len(argv) < 2:
        operate_mode = '0'
    else:
        operate_mode = str(argv[1])
    return operate_mode


# load training data
data_train, label_train = load.load_data_set('../../data/a1_svm/train.csv')
# load testing data
data_test, label_test = load.load_data_set('../../data/a1_svm/test.csv')


# train primal and dual models in different mode
mode = main(sys.argv)
#svm_model = np.zeros((201, 1))
#svm_model_d = np.zeros((201, 1))

if mode not in ['0', '1', '2']:
    print("\nPlease choose a correct operation mode!"
          "\nConsole command: python3 main.py <mode>"
          "\n                 mode = {0, 1, 2}")
else:
    if mode != '2':
        # mode 0: completely run the whole experiment
        if mode == '0':
            # 0. Find optimal C by cross validation
            print('\n# 0. Cross validation: decide optimal C for primal & dual SVM\n')
            Cs = [1, 5, 10, 100]
            regularisation_para_C = validate.cross_validate(data_train, label_train, 5, Cs)
        # mode 1: skip cross validation, use optimal C directly
        else:
            regularisation_para_C = 5

        # 1. Implement Primal SVM and predict
        # train svm in the primal
        print('\n\n# 1. Primal: Implement SVM')
        svm_model = primal.svm_train_primal(data_train, label_train, regularisation_para_C)

        # 2. Implement Dual SVM and predict
        # train svm in the dual
        print('\n\n# 2. Dual: Implement SVM')
        svm_model_d = dual.svm_train_dual(data_train, label_train, regularisation_para_C)

    # mode 2: will not train the model, just load model to predict
    if mode == '2':
        print('\n\n# 1. Primal: Load model')
        svm_model = np.loadtxt('../../output/svm_model_primal', dtype=float, delimiter=',')
        svm_model = svm_model.reshape(svm_model.size, 1)
        print('\n\n# 2. Dual: Load model')
        svm_model_d = np.loadtxt('../../output/svm_model_primal', dtype=float, delimiter=',')
        svm_model_d = svm_model_d.reshape(svm_model_d.size, 1)


    # 3. use sklearn implementations for comparison
    # train & predict with sklearn
    print('\n\n# 3. Sklearn: train')
    svm_model_sk, sk_classifier = sk.svm_sk(data_train, label_train, mode)

    # 4. predict and compute accuracy of svm_model
    print('\n\n# 4. Primal: predict')
    print('\n## Primal: Predict with training data set:')
    train_accuracy = predict.svm_predict_primal(data_train, label_train, svm_model)
    print('     accuracy: %.2f%%' % (100 * train_accuracy))
    print('## Primal: Predict with testing data set:')
    test_accuracy = predict.svm_predict_primal(data_test, label_test, svm_model)
    print('     accuracy: %.2f%%' % (100 * test_accuracy))

    # 5. predict and compute accuracy of svm_model_d
    print('\n\n# 5. Dual: predict')
    print('\n## Dual: Predict with training data set:')
    train_accuracy_d = predict.svm_predict_primal(data_train, label_train, svm_model_d)
    print('     accuracy(dual): %.2f%%' % (100 * train_accuracy_d))
    print('## Dual: Predict with testing data set:')
    test_accuracy_d = predict.svm_predict_dual(data_test, label_test, svm_model_d)
    print('     accuracy(dual): %.2f%%' % (100 * test_accuracy_d))

    # 6. print accuracy of svm_model_sk
    print('\n\n# 6. Sklearn: predict')
    print('\n## Sklearn: Predict with training data set:')
    print('     accuracy： %.2f%%' % (100 * svm_model_sk.score(data_train, label_train)))
    print('## Sklearn: Predict with testing data set:')
    print('     accuracy： %.2f%%' % (100 * svm_model_sk.score(data_test, label_test)))

    # 7. compare results
    print('\n\n# 7. Compare solutions')
    compare.compare_svm(svm_model, svm_model_d, "primal", "dual")
    compare.compare_svm(svm_model, sk_classifier, "primal", "sklearn")

print("\n")
