# Author: Wuli Zuo, a1785343
# Date: 2020-10-24


import matplotlib.pyplot as plt


# plot accuracy curve against reduced dimensions
def plot_acc(dim, acc, title):
    plt.figure(figsize=(8, 4))
    plt.title('%s against reduced dimensions' % title)
    plt.xlabel('dimension')
    plt.ylabel('accuracy: %')

    y = 100 * acc
    plt.xticks(dim, dim)
    plt.xlim(dim[0], dim[len(dim) - 1])

    plt.plot(dim, y, label='%s acc' % title, marker='o')
    for a, b in zip(dim, y):
        plt.text(a, b, '%.2f%%' % b, va='bottom', ha='left', fontsize=10)
    plt.legend(loc='best')

    plt.savefig('../../output/%s_acc&dim.png' % title)
    plt.show()


# plot multiple accuracies against reduced dimensions
def plot_accs(dim, acc1, acc2, class1, class2, title):
    plt.figure(figsize=(8, 4))
    plt.title('%s against reduced dimensions' % title)
    plt.xlabel('dimension')
    plt.ylabel('accuracy: %')

    y1 = 100 * acc1
    y2 = 100 * acc2
    plt.xticks(dim, dim)
    plt.xlim(dim[0], dim[len(dim) - 1])

    plt.plot(dim, y1, color='b', label='%s acc' % class1, marker='o')
    for a, b in zip(dim, y1):
        plt.text(a, b, '%.2f%%' % b, ha='left', va='bottom', fontsize=8)
    plt.plot(dim, y2, color='r', label='%s acc' % class2, marker='o')
    for a, b in zip(dim, y2):
        plt.text(a, b, '%.2f%%' % b, ha='left', va='top', fontsize=8)
    plt.legend(loc='lower left')

    plt.savefig('../../output/%s&dim.png' % title)
    plt.show()


# plot loss curve against rounds of iterations
def plot_loss(loss, title):
    plt.figure(figsize=(8, 4))
    plt.title('%s loss against iterations' % title)
    plt.xlabel('# round')
    plt.ylabel('loss')

    for i in range(len(loss)):
        x = range(0, len(loss[i]))
        plt.plot(x, loss[i], label='Experiment #%d %s loss' % (i, title))
    plt.legend(loc='best')

    plt.savefig('../../output/loss&round.png')
    plt.show()


# plot percentages against reduced dimensions
def plot_pcts(dim, percentages, title):
    plt.figure(figsize=(8, 4))
    plt.title('%s against reduced dimensions' % title)
    plt.xlabel('dimension')
    plt.ylabel('avg_percentage: %')

    plt.xticks(dim, dim)
    plt.xlim(dim[0], dim[len(dim) - 1])

    for i in range(percentages.shape[0]):
        y = 100 * percentages[i]
        plt.plot(dim, y, label='experiment #%d' % i, marker='o')
        for a, b in zip(dim, y):
            plt.text(a, b, '%.2f%%' % b, ha='left', va='center', fontsize=10)
    plt.legend(loc='best')

    plt.savefig('../../output/%s&dim.png' % title)
    plt.show()
