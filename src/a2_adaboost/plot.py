# Author: Wuli Zuo, a1785343
# Date: 2020-09-12


import matplotlib.pyplot as plt


# plot the relationship between error and the number of weak classifiers
def plot_acc_round(acc1, acc2, name1, name2, title):
    plt.figure(figsize=(8, 4))
    plt.title('Accuracy of different rounds of iterations-%s' % title)
    plt.xlabel('# rounds')
    plt.ylabel('accuracy: %')

    x = range(0, acc1.shape[0])
    y1 = 100 * acc1
    y2 = 100 * acc2
    max_idx_1 = y1.argmax()
    max_idx_2 = y2.argmax()

    plt.plot(x, y1, color='b', label=name1+' accuracy')
    plt.scatter(max_idx_1, y1[max_idx_1], color='b', label='Max %s accuracy: %.2f%% at %d' % (name1, y1[max_idx_1], max_idx_1))
    plt.plot(x, y2, color='r', label=name2+' accuracy')
    plt.scatter(max_idx_2, y2[max_idx_2], color='r', label='Max %s accuracy: %.2f%% at %d' % (name2, y2[max_idx_2], max_idx_2))
    plt.legend()

    plt.savefig('../../output/acc&iteration-%s.png' % title)
    plt.show()


def plot_acc_round_compare(acc1, acc2, acc3, acc4, name1, name2, title1, title2):
    plt.figure(figsize=(8, 4))
    plt.title('Accuracy of different rounds of iterations-%s vs %s' % (title1, title2) )
    plt.xlabel('# rounds')
    plt.ylabel('accuracy: %')

    x = range(0, acc1.shape[0])
    y1 = 100 * acc1
    y2 = 100 * acc2
    y3 = 100 * acc3
    y4 = 100 * acc4
    max_idx_1 = y1.argmax()
    max_idx_2 = y2.argmax()
    max_idx_3 = y3.argmax()
    max_idx_4 = y4.argmax()

    plt.plot(x, y1, color='b', label=title1+' '+name1+' accuracy')
    plt.scatter(max_idx_1, y1[max_idx_1], color='b',
                label='Max accuracy: %.2f%% at %d' % (y1[max_idx_1], max_idx_1))
    plt.plot(x, y2, color='r', label=title1+' '+name2+' accuracy')
    plt.scatter(max_idx_2, y2[max_idx_2], color='r',
                label='Max accuracy: %.2f%% at %d' % (y2[max_idx_2], max_idx_2))
    plt.plot(x, y3, color='g', label=title2+' '+name1+' accuracy')
    plt.scatter(max_idx_3, y3[max_idx_3], color='g',
                label='Max accuracy: %.2f%% at %d' % (y3[max_idx_3], max_idx_3))
    plt.plot(x, y4, color='orange', label=title2+' '+name2+' accuracy')
    plt.scatter(max_idx_4, y4[max_idx_4], color='orange',
                label='Max accuracy: %.2f%% at %d' % (y4[max_idx_4], max_idx_4))

    plt.legend()

    plt.savefig('../../output/acc&iteration-%s vs %s.png' % (title1, title2))
    plt.show()
