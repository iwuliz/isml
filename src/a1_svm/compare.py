# Author: Wuli Zuo, a1785343
# Date: 2020-08-27 23:43:12

import numpy as np
import matplotlib.pyplot as plt


# function of comparing results of SVMs
def compare_svm(model1, model2, name1, name2):
    m = model1.size - 1
    w1 = model1[:m]
    b1 = model1[m]
    w2 = model2[:m]
    b2 = model2[m]
    w_diff = np.abs(w1 - w2)
    w_diff_pct = np.abs(100 * w_diff / w1)
    b_diff = abs(b1 - b2)

    # draw scatter of w values
    plt.figure(figsize=(8, 4))
    plt.title('w values of %s SVM and %s SVM' % (name1, name2))
    plt.xlabel('i')
    plt.ylabel('w[i]')

    x_axis = np.array([i for i in range(0, m)])
    plt.scatter(x_axis, [w for w in w1.squeeze()], s=10, c='r', alpha=0.5, label=name1+' w values')
    plt.scatter(x_axis, [w for w in w2.squeeze()], s=10, c='b', alpha=0.5, label=name2+' w values')

    plt.xlim([-1, 200])
    plt.legend()
    plt.savefig('../../output/w_values_%s&%s.png' % (name1, name2))
    plt.show()

    # draw distribution of difference of w
    plt.figure(figsize=(8, 4))
    plt.title('Difference of w between %s SVM and %s SVM' % (name1, name2))
    plt.xlabel('w[i]')
    plt.ylabel('w_diff_pct=abs(1-w2/w1): %')

    x_axis = np.array([i for i in range(0, m)])
    plt.bar(x_axis, [i for i in w_diff_pct.squeeze()], color='bgr')

    plt.xlim([-1, 200])
    plt.savefig('../../output/w_diff_%s&%s.png' % (name1, name2))
    plt.show()

    # print difference
    print('\n## Difference between %s and %s' % (name1, name2))
    print('     w_diff: in range (%.8f, %.8f)' % (w_diff.min(), w_diff.max()))
    print('     w_diff_pct: in range (%.8f%%, %.8f%%)' % (w_diff_pct.min(), w_diff_pct.max()))
    print('                 mean = %.8f%%' % w_diff_pct.mean())
    print('                 std  = %.8f%%' % w_diff.std())
    print('     b_diff: %.8f (%.8f%%)' % (b_diff, 100 * b_diff / b1))
