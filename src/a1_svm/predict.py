# Author: Wuli Zuo, a1785343
# Date: 2020-08-19 11:05:08

def svm_model_predict(data, label, model):
    m = len(data[0])  # features
    n = len(label)  # samples / labels
    mistake = 0
    support = 0
    accurate = 0
    w = model[:m]
    b = model[m]
    predict = []

    for i in range(n-1):
        y = 0
        for j in range(m-2):
            y += w[j]*data[i][j]
        y += b
        if label[i][0]*y < 0:
            mistake += 1
        else:
            if label[i][0]*y == 0.00:
                support += 1
            else:
                accurate += 1
        predict.append(y)
    '''
    print("mistake = ", mistake)
    print("support = ", support)
    print("accurate = ", accurate)
    print(np.array(predict)) # output predict results
    '''
    return 1-mistake/n

# function of testing with svm_model
def svm_predict_primal (data_test, label_test, svm_model):
    return svm_model_predict(data_test, label_test, svm_model)

# function of testing with svm_model_d
def svm_predict_dual(data_test, label_test, svm_model_d):
    return svm_model_predict(data_test , label_test, svm_model_d)

