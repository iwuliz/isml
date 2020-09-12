# Author: Wuli Zuo, a1785343
# Date: 2020-09-12


import matplotlib.pyplot as plt


# plot the relationship between error and the number of weak classifiers
def plot_err_round(err, name):
    plt.figure(figsize=(8, 4))
    plt.title('Error rate of different iterations for %s data ' % name)
    plt.xlabel('iteration')
    plt.ylabel('error rate: %')

    x = range(100)
    y = 100 * err
    min_idx = y.argmin()
    plt.plot(x, y)
    plt.scatter(min_idx, y[min_idx], label='Min error: %.2f%% at %d' % (y[min_idx], min_idx))
    plt.legend()

    plt.savefig('../../output/%s_data_err.png' % name)
    plt.show()
