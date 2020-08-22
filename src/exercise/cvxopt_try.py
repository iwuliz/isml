# https://blog.csdn.net/guyanbeifei/article/details/79691576
#    优化中传递的matrix必须是cvxopt内的matrix
#    matrix()  转换对应的类型为numpy.array，numpy.matrix可能也行（没有尝试）


import numpy as np
from cvxopt import solvers
from cvxopt import matrix as mx
import matplotlib
import matplotlib.pyplot as plt

def loadDataSet(fileName):
    dataMat = []; labelMat = []
    fr = open(fileName)
    for line in fr.readlines():
        lineArr = line.strip().split('\t') #用tab分隔，lineArr[0] 是x1，lineArr[1] 是x2，lineArr[2] 是y
        dataMat.append([float(lineArr[0]), float(lineArr[1])]) #生成features的列表
        labelMat.append(float(lineArr[2])) #生成label的列表
    return dataMat,labelMat

data,label = loadDataSet('testSet.txt')
dataLen = len(data[0]) #dataLen是featrues的个数

# 生成cvxopt里面对应的参数矩阵/向量
p1 = np.zeros((1, dataLen+1)) #新建一个1*(dateLen+1)的矩阵，以0填充
p2 = np.zeros((dataLen,1)) #新建一个(dateLen+1)*1的矩阵，以0填充
p3 = np.eye(dataLen) #生成对角矩阵，默认对角线填充1
ptmp = np.hstack((p2,p3)) #横向拼接p2, p3矩阵
p = np.vstack((p1,ptmp)) #纵向拼接p1, ptmpt矩阵
p = mx(p)

q = mx(np.zeros((dataLen+1,1)))

m = len(label)
#dataT = np.array(my_a1)
a = []
for i in range(m):
    tmp = [1]+data[i]
    for j in range(dataLen+1):
        tmp[j] = -1*label[i]*tmp[j]
    a.append(tmp)
G = mx(np.array(a))

h = mx(np.zeros((m,1))-1)

# 求解凸二次规划问题
sol = solvers.qp(p,q,G,h)

# 打印信息
print(sol['status'])
print(sol['x'])

# 绘制数据散点图
datanp = np.array(data)
labelnp = np.array(label)
data1 = datanp[labelnp==1]
data2 = datanp[labelnp==-1]

f1 = plt.figure(1)
plt.scatter(data1[:,0],data1[:,1], marker = 'x', color = 'm', label='1', s = 30)
plt.scatter(data2[:,0],data2[:,1], marker = 'o', color = 'r', label='-1', s = 15)

# 打印分割平面
b = sol['x'][0]
w1 = sol['x'][1]
w2 = sol['x'][2]

xmin = min(datanp[:,0])
xmax = max(datanp[:,0])
x2_1 = -1.0*(b+w1*xmin)/w2
x2_2 = -1.0*(b+w1*xmax)/w2

plt.plot([xmin,xmax],[x2_1,x2_2])
plt.show()