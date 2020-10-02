# Author: Wuli Zuo, a1785343
# Date: 2020-09-12


import matplotlib.pyplot as plt


# plot the relationship between error and the number of weak classifiers
def plot_err_round(err1, err2, name1, name2, title):
    plt.figure(figsize=(8, 4))
    plt.title('Error rate of different rounds of iterations-%s' % title)
    plt.xlabel('# rounds')
    plt.ylabel('error rate: %')

    x = range(0, err1.shape[0])
    y1 = 100 * err1
    y2 = 100 * err2
    min_idx_1 = y1.argmin()
    min_idx_2 = y2.argmin()

    plt.plot(x, y1, color='b', label=name1+' error')
    plt.scatter(min_idx_1, y1[min_idx_1], color='b', label='Min %s error: %.2f%% at %d' % (name1, y1[min_idx_1], min_idx_1))
    plt.plot(x, y2, color='r', label=name2+' error')
    plt.scatter(min_idx_2, y2[min_idx_2], color='r', label='Min %s error: %.2f%% at %d' % (name2, y2[min_idx_2], min_idx_2))
    plt.legend()

    plt.savefig('../../output/err&iteration-%s.png' % title)
    plt.show()
