# Author: Wuli Zuo, a1785343
# Date: 2020-10-24


import kmeans
import load
import noise
import numpy as np
import pca
import plot
import random
import sk
import time


start = time.time()

# load data
data_train, label_train = load.load_data_set('../../data/a3_pca/mnist_train.csv')
data_test, label_test = load.load_data_set('../../data/a3_pca/mnist_test.csv')


# Task 1: implement PCA

# generate eigenvalue and eigenvector of the training data
print('# 1 Implement PCA')
eigen_val, eigen_vec = pca.eigen(data_train)
print('\n## eigen_val:\n', eigen_val)


# Task 2: apply PCA on both training data and test data,
#         perform classification with the 1-nearest neighbour classifier
print('\n# 2 Apply PCA on training and test data, classify with 1NN')

# apply PCA and perform 1NN for different reduced dimensions
print('\n## Accuracy of different reduced dimensions:')
print('   dim \t  1NN with my PCA\t\t  1NN with Sk PCA\t\t\t     SVM')
print('       \ttrain acc\ttest acc\ttrain acc\ttest acc\ttrain acc\ttest acc')

P_train = pca.pca_reconstruct(data_train, eigen_vec)
P_test = pca.pca_reconstruct(data_test, eigen_vec)
dim_options = [256, 128, 64, 32, 16, 8, 4, 2]
acc_test_1nn = []
test_acc_svm = []
for dim in dim_options:
    # apply PCA on training data and test data
    select_data_train_my = pca.pca_apply(P_train, dim)
    select_data_test_my = pca.pca_apply(P_test, dim)

    # apply sklearn PCA for comparison
    select_data_train_sk, select_data_test_sk = sk.sk_pca(data_train, data_test, dim)

    # test my PCA and sklearn PCA with 1NN
    acc_train_my_d, acc_test_my_d = sk.sk_1nn(select_data_train_my, label_train, select_data_test_my, label_test)
    acc_test_1nn.append(acc_test_my_d)
    acc_train_sk_d, acc_test_sk_d = sk.sk_1nn(select_data_train_sk, label_train, select_data_test_sk, label_test)

    # test my PCA with sklearn SVM
    sk_svm = sk.svm_sk(select_data_train_my, label_train)
    acc_train_svm_d = sk_svm.score(select_data_train_my, label_train)
    acc_test_svm_d = sk_svm.score(select_data_test_my, label_test)
    test_acc_svm.append(acc_test_svm_d)

    print('   %d \t%.2f%% \t%.2f%% \t\t%.2f%% \t%.2f%% \t\t%.2f%% \t\t%.2f%%' %
          (dim, float(acc_train_my_d*100), float(acc_test_my_d*100),
           float(acc_train_sk_d*100), float(acc_test_sk_d*100),
           float(acc_train_svm_d*100), float(acc_test_svm_d*100)))

# plot error curve against reduced dimensions
plot.plot_acc(dim_options, np.array(acc_test_1nn), 'PCA&1NN')
plot.plot_accs(dim_options, np.array(acc_test_1nn), np.array(test_acc_svm), '1NN', 'SVM', 'Accuracy')


# Task 3: classify training data with K-means
print('\n# 3 Implement K-means clustering')

data_set = data_train.tolist()
loss_list = []
# repeat this experiment for three times to reduce bias of random:
for i in range(3):
    # randomly select 10 samples as the initial centres
    centres = random.sample(data_set, 10)

    # classify with K-means
    centres, clusters, clusters_idx, loss_list_i = kmeans.kmeans(data_set, centres, 10)
    loss_list.append(loss_list_i)
    print('\n## Exp %d\n   loss: %s' %(i, loss_list_i))
    print('   final loss: ', loss_list_i[len(loss_list_i)-1])

# plot loss curve against iterations
plot.plot_loss(loss_list, 'K-means')


# Task 4: apply K-means with fixed random initial centres to reduced dimensions
print('\n# 4 Perform K-means with fixed random initial centres')
print('\n## Average percentage of different reduced dimensions:')

dim_options = [256, 128, 64, 32, 16, 8, 4, 2]
pcts = []
# repeat this experiment for three times to reduce bias of random:
for i in range(3):
    # randomly select initial centres of classes (0,9)
    init_idx_list, init_labels = kmeans.random_initial(label_train, 10)
    # apply PCA and K-means for different reduced dimensions
    pcts.append([])
    print('   Exp %d running...' % (len(pcts)-1))
    for dim in dim_options:
        # apply PCA on training data and centres
        select_data_train = pca.pca_apply(P_train, dim)
        centres = select_data_train[init_idx_list]
        centres = centres.tolist()
        # perform K-means
        data_set = select_data_train.tolist()
        centres, clusters, clusters_idx, loss_list = kmeans.kmeans(data_set, centres, 10)
        pct = kmeans.cal_pct(init_idx_list, clusters_idx, label_train)
        pcts[i].append(pct)

print('\n   Dim \t\tAvg_pct 0 \t\tAvg_pct 1 \t\tAvg_pct 2 ')
for j in range(len(dim_options)):
    print('   %d \t\t%.2f%%  \t\t%.2f%%  \t\t%.2f%%' %
          (dim_options[j], float(pcts[0][j]*100), float(pcts[1][j]*100), float(pcts[2][j]*100)))

# plot average percentage curve against the number of reduced dimensions
plot.plot_pcts(dim_options, np.array(pcts), 'Avg_percentage')


# Task 5: append noisy dimensions, apply PCA, test and analyse with 1NN and SVM
print('\n# 5 Append noisy dimensions, test and analyse with 1NN and SVM')

# append Gaussian noise to training data and test data
data_train_noised = noise.add_gauss(data_train, 256)
data_test_noised = noise.add_gauss(data_test, 256)

# compare before and after adding noise, test with 1NN and SVM
print('\n## Accuracy on clean and noised data:')
# train and predict with 1NN on clean data and noised data
acc_train_1nn, acc_test_1nn = sk.sk_1nn(data_train, label_train, data_test, label_test)
acc_train_1nn_noised, acc_test_1nn_noised = sk.sk_1nn(data_train_noised, label_train, data_test_noised, label_test)
# train and predict with SVM on clean data and noised data
sk_svm = sk.svm_sk(data_train, label_train)
acc_train_svm = sk_svm.score(data_train, label_train)
acc_test_svm = sk_svm.score(data_test, label_test)
sk_svm = sk.svm_sk(data_train_noised, label_train)
acc_train_svm_noised = sk_svm.score(data_train_noised, label_train)
acc_test_svm_noised = sk_svm.score(data_test_noised, label_test)
print('   \t\tclean train acc\tclean test acc\tnoised train acc\tnoised test acc')
print('   %s\t\t%.2f%% \t\t%.2f%% \t\t%.2f%% \t\t%.2f%% ' %
      ('1NN', float(acc_train_1nn * 100), float(acc_test_1nn * 100),
       float(acc_train_1nn_noised * 100), float(acc_test_1nn_noised * 100)))
print('   %s\t\t%.2f%% \t\t%.2f%% \t\t%.2f%% \t\t%.2f%% ' %
      ('SVM', float(acc_train_svm * 100), float(acc_test_svm * 100),
       float(acc_train_svm_noised * 100), float(acc_test_svm_noised * 100)))

# generate eigenvalue and eigenvector of the polluted data
eigen_val_noised, eigen_vec_noised = pca.eigen(data_train_noised)

# apply PCA, perform 1NN and SVM for different reduced dimensions
print('\n## Accuracy (noised) of different reduced dimensions:')
print('   dim \t\t1NN train acc\t1NN test acc\tSVM train acc\tSVM test acc')

P_train_noised = pca.pca_reconstruct(data_train_noised, eigen_vec_noised)
P_test_noised = pca.pca_reconstruct(data_test_noised, eigen_vec_noised)
dim_options = [256, 128, 64, 32, 16, 8, 4, 2]
acc_test_1nn = []
test_acc_svm = []
for dim in dim_options:
    # apply PCA on polluted training data and test data
    select_data_train_noised = pca.pca_apply(P_train_noised, dim)
    select_data_test_noised = pca.pca_apply(P_test_noised, dim)

    # test with 1NN
    acc_train_my_d, acc_test_my_d = sk.sk_1nn(select_data_train_noised, label_train,
                                              select_data_test_noised, label_test)
    acc_test_1nn.append(acc_test_my_d)
    
    # test with SVM
    sk_svm = sk.svm_sk(select_data_train_noised, label_train)
    acc_train_svm_d = sk_svm.score(select_data_train_noised, label_train)
    acc_test_svm_d = sk_svm.score(select_data_test_noised, label_test)
    test_acc_svm.append(acc_test_svm_d)
    
    print('   %d \t\t%.2f%% \t\t%.2f%% \t\t\t%.2f%% \t\t%.2f%%' %
          (dim, float(acc_train_my_d*100), float(acc_test_my_d * 100),
           float(acc_train_svm_d*100), float(acc_test_svm_d*100)))

# plot error curve against reduced dimensions
plot.plot_accs(dim_options, np.array(acc_test_1nn), np.array(test_acc_svm), '1NN', 'SVM', 'Accuracy_noised')


end = time.time()
print('\nRunning time: %ss' % (end-start))
