#Author: Wuli Zuo, a1785343
#date: 2020-08-18 16:17:22

import sklearn
from sklearn import svm
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


#define converts(字典)
def Iris_label(s):
    it={b'Iris-setosa':0, b'Iris-versicolor':1, b'Iris-virginica':2 }
    return it[s]

#1.读取数据集
path= '../../data/iris/iris.data'  # my_a1 的存储位置
data=np.loadtxt(path, dtype=float, delimiter=',', converters={4:Iris_label} ) #使用loadtxt读取path所指向的数据文件
#converters={4:Iris_label}中“4”指的是第5列：将第5列的str转化为label(number)

#2.划分数据与标签
x,y=np.split(data,indices_or_sections=(4,),axis=1) #x为数据，y为标签
x=x[:,0:2] #为便于后边画图显示，只选取前两维度。若不用画图，可选取前四列x[:,0:4]
train_data,test_data,train_label,test_label =sklearn.model_selection.train_test_split(x,y, random_state=1, train_size=0.6,test_size=0.4)

#3.训练svm分类器
#kernel='linear'时，为线性核，C越大分类效果越好，但有可能会过拟合（defaul C=1）。
#kernel='rbf'时（default），为高斯核，gamma值越小，分类界面越连续；gamma值越大，分类界面越“散”，分类效果越好，但有可能会过拟合。
# decision_function_shape='ovr'时，为one v rest（一对多），即一个类别与其他类别进行划分，
# decision_function_shape='ovo'时，为one v one（一对一），即将类别两两之间进行划分，用二分类的方法模拟多分类的结果。
classifier=svm.SVC(C=2,kernel='rbf',gamma=10,decision_function_shape='ovr') # ovr:一对多策略
classifier.fit(train_data,train_label.ravel()) #ravel函数在降维时默认是行序优先

#4.计算svc分类器的准确率
print("Trainng Set：",classifier.score(train_data,train_label))
print("Testing set：",classifier.score(test_data,test_label))

#也可直接调用accuracy_score方法计算准确率
from sklearn.metrics import accuracy_score
tra_label=classifier.predict(train_data) #训练集的预测标签
tes_label=classifier.predict(test_data) #测试集的预测标签
print("Trainging Set：", accuracy_score(train_label,tra_label) )
print("Testing Set：", accuracy_score(test_label,tes_label) )

#查看决策函数
print('train_decision_function:\n',classifier.decision_function(train_data)) # (90,3)
print('predict_result:\n',classifier.predict(train_data))

# 5.绘制图形
# 确定坐标轴范围
x1_min, x1_max = x[:, 0].min(), x[:, 0].max()  # 第0维特征的范围
x2_min, x2_max = x[:, 1].min(), x[:, 1].max()  # 第1维特征的范围
x1, x2 = np.mgrid[x1_min:x1_max:200j, x2_min:x2_max:200j]  # 生成网络采样点
grid_test = np.stack((x1.flat, x2.flat), axis=1)  # 测试点

print(x1.shape)

# 指定默认字体
matplotlib.rcParams['font.sans-serif'] = ['SimHei']
# 设置颜色
cm_light = matplotlib.colors.ListedColormap(['#A0FFA0', '#FFA0A0', '#A0A0FF'])
cm_dark = matplotlib.colors.ListedColormap(['g', 'r', 'b'])
grid_hat = classifier.predict(grid_test)  # 预测分类值
grid_hat = grid_hat.reshape(x1.shape)  # 使之与输入的形状相同

plt.pcolormesh(x1, x2, grid_hat, cmap=cm_light)     # 预测值的显示
plt.scatter(x[:, 0], x[:, 1], c=y[:,0], s=30,cmap=cm_dark)  # 样本
plt.scatter(test_data[:,0],test_data[:,1], c=test_label[:,0],s=30,edgecolors='k', zorder=2,cmap=cm_dark) #圈中测试集样本点
plt.xlabel('calyx length', fontsize=13)
plt.ylabel('calyx width', fontsize=13)
plt.xlim(x1_min,x1_max)
plt.ylim(x2_min,x2_max)
plt.title('Iris SVM binary classification')
plt.show()