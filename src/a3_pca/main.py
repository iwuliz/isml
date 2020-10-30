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
print('#1 Implement PCA')
eigen_val, eigen_vec, index = pca.eigen(data_train)
print('\n## eigen_val:\n', eigen_val)


# Task 2: apply PCA on both training data and test data,
#         perform classification with the 1-nearest neighbour classifier
print('\n#2 Apply PCA on training and test data, classify with KNN')

# apply PCA and k-means for different reduced dimensions
print('\n## Accuracy of different reduced dimensions:')
print('   dim \tMy train acc\tMy test acc\t Sk train acc\tSk test acc\tSVM train acc\tSVM test acc')
dim_options = [784, 256, 128, 64, 32, 10]
acc_test_pca = []
test_acc_svm = []
for dim in dim_options:
    # apply PCA on training data and test data
    select_data_train_my = pca.pca(data_train, eigen_vec, index, dim)
    select_data_test_my = pca.pca(data_test, eigen_vec, index, dim)

    # apply sklearn PCA for comparison
    select_data_train_sk, select_data_test_sk = sk.sk_pca(data_train, data_test, dim)

    # test my PCA and sklearn PCA with KNN, set k = 1
    acc_train_my_d, acc_test_my_d = sk.sk_knn(select_data_train_my, label_train, select_data_test_my, label_test, dim)
    acc_test_pca.append(acc_test_my_d)
    acc_train_sk_d, acc_test_sk_d = sk.sk_knn(select_data_train_sk, label_train, select_data_test_sk, label_test, dim)

    # test my PCA with sklearn SVM
    sk_svm = sk.svm_sk(select_data_train_my, label_train)
    acc_train_svm_d = sk_svm.score(select_data_train_my, label_train)
    acc_test_svm_d = sk_svm.score(select_data_test_my, label_test)
    test_acc_svm.append(acc_test_svm_d)

    print('   %d \t\t%.2f%% \t\t%.2f%% \t\t%.2f%% \t\t%.2f%% \t\t%.2f%% \t\t%.2f%%' %
          (dim, float(acc_train_my_d*100), float(acc_test_my_d*100),
           float(acc_train_sk_d*100), float(acc_test_sk_d*100),
           float(acc_train_svm_d*100), float(acc_test_svm_d*100)))

# plot error curve against reduced dimensions
plot.plot_acc(dim_options, np.array(acc_test_pca), 'PCA')
plot.plot_accs(dim_options, np.array(acc_test_pca), np.array(test_acc_svm), 'PCA', 'SVM', 'Accuracy')


# Task 3: classify training data with k-means
print('\n#3 Implement k-means clustering')

data_set = data_train.tolist()
loss_list = []
# repeat this experiment for three times to reduce bias of random:
for i in range(3):
    # randomly select 10 samples as the initial centroids
    centroids = random.sample(data_set, 10)

    # classify with k-means
    centroids, clusters, clusters_idx, loss_list_i = kmeans.kmeans(data_set, centroids, 10)
    loss_list.append(loss_list_i)
    print('\n## Exp %d\n   loss: %s' %(i, loss_list_i))
    print('   final loss: ', loss_list_i[len(loss_list_i)-1])

# plot loss curve against iterations
plot.plot_loss(loss_list, 'K-means')


# Task 4: apply k-means with fixed random initial centres to reduced dimensions
print('\n#4 Perform k-means with fixed random initial centres')
print('\n## Average percentage of different reduced dimensions:')

dim_options = [784, 256, 128, 64, 32, 10]
pcts = []
# repeat this experiment for three times to reduce bias of random:
for i in range(3):
    # randomly select initial centres of classes (0,9)
    init_idx_list, init_labels = kmeans.random_initial(label_train, 10)

    # apply PCA and k-means for different reduced dimensions
    pcts.append([])
    for dim in dim_options:
        # apply PCA on training data and centres
        select_data_train_my = pca.pca(data_train, eigen_vec, index, dim)
        centroids = select_data_train_my[init_idx_list]
        centroids = centroids.tolist()
        # perform k-means
        data_set = select_data_train_my.tolist()
        centroids, clusters, clusters_idx, loss_list = kmeans.kmeans(data_set, centroids, 10)
        pct = kmeans.cal_pct(init_idx_list, clusters_idx, label_train)

        pcts[i].append(pct)
print('   Dim \tAvg_pct 1 \t\tAvg_pct 2 \t\tAvg_pct 3 ')
for j in range(len(dim_options)):
    print('   %d \t\t%.2f%%  \t\t%.2f%%  \t\t%.2f%%' %
          (dim_options[j], float(pcts[0][j]*100), float(pcts[1][j]*100), float(pcts[2][j]*100)))

# plot average percentage curve against the number of reduced dimensions
plot.plot_pcts(dim_options, np.array(pcts), 'Avg_percentage')


# Task 5: append noisy dimensions, test and analyse with PCA&KNN and SVM
print('\n#5 Append noisy dimensions, test and analyse with PCA&KNN and SVM')

# append Gaussian noise to training data and test data
data_train_polluted = noise.add_gauss(data_train, 256)
data_test_polluted = noise.add_gauss(data_test, 256)

# generate eigenvalue and eigenvector of the polluted data
eigen_val, eigen_vec, index = pca.eigen(data_train_polluted)

# apply PCA PCA&k-means and SVM for different reduced dimensions
print('\n## Accuracy (noised) of different reduced dimensions:')
print('   dim \t\tPCA train acc\tPCA test acc\tSVM train acc\tSVM test acc')
dim_options = [1040, 256, 128, 64, 32, 10]
acc_test_pca = []
test_acc_svm = []
for dim in dim_options:
    # apply PCA on polluted training data and test data
    select_data_train_my = pca.pca(data_train_polluted, eigen_vec, index, dim)
    select_data_test_my = pca.pca(data_test_polluted, eigen_vec, index, dim)

    # test with KNN
    acc_train_my_d, acc_test_my_d = sk.sk_knn(select_data_train_my, label_train, select_data_test_my, label_test, dim)
    acc_test_pca.append(acc_test_my_d)
    
    # test with SVM
    sk_svm = sk.svm_sk(data_train_polluted, label_train)
    acc_train_svm_d = sk_svm.score(data_train_polluted, label_train)
    acc_test_svm_d = sk_svm.score(data_test_polluted, label_test)
    test_acc_svm.append(acc_test_svm_d)
    
    print('   %d \t\t%.2f%% \t\t%.2f%% \t\t%.2f%% \t\t%.2f%%' %
          (dim, float(acc_train_my_d*100), float(acc_test_my_d * 100),
           float(acc_train_svm_d*100), float(acc_test_svm_d*100)))

# plot error curve against reduced dimensions
plot.plot_accs(dim_options, np.array(acc_test_pca), np.array(test_acc_svm), 'PCA', 'SVM', 'Accuracy_noised')


end = time.time()
print('running time: %ss' % (end-start))