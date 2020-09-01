# Author: Wuli Zuo, a1785343
# Date: 2020-08-27 23:05:54

import numpy as np
import matplotlib.pyplot as plt


def draw_C_acc(C, acc, name):
    plt.figure(figsize=(8, 4))
    plt.title('Accuracy of different Cs for the %s SVM' % name)
    plt.xlabel('C')
    plt.ylabel('cross validation accuracy: %')

    x_scale = range(len(C))
    y = 100*np.array(acc)

    plt.xticks(x_scale, C)

    plt.scatter(x_scale, y, label='accuracy')
    plt.plot(x_scale, y)

    for a, b in zip(x_scale, y):
        plt.text(a, b, '%.2f%%' % b, ha='center', va='bottom', fontsize=10)

    plt.savefig('../../output/C_%s.png' % name)
    plt.show()
