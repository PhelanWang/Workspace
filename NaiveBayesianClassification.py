# coding: utf-8

import numpy as np
import matplotlib.pyplot as plt
"""
每个样本两个特征值，假设每个特征值都服从正太分布
每个样本被分类到0或者1
"""


def normal_distribution(x, mean, var):
    return 1/(np.sqrt(2*np.pi*var))*np.power(np.e, -(mean-x)**2/(2*var))


data = np.loadtxt('./data/logistic_data.txt', delimiter=',')

X = data[:70, 0:2]
Y = data[:70, 2:3]

TestX = data[70:, 0:2]
TestY = data[70:, 2:3]

# 40 20的概率
t0 = 40
t1 = 20

# 分别找出分类为0和分类为1的样本
X_Y0 = X[np.where(Y == 0)[0], :]
X_Y1 = X[np.where(Y == 1)[0], :]


# 计算分类为0和分类为1的样本
P_Y0 = float(X_Y0[:, 0].size)/Y.size
P_Y1 = float(X_Y1[:, 1].size)/Y.size


# 计算在分类0下对于x0和x1的概率
X0_Y0 = normal_distribution(40, X_Y0[:, 0].mean(), X_Y0[:, 0].var())
X1_Y0 = normal_distribution(20, X_Y0[:, 1].mean(), X_Y0[:, 1].var())
result0 = P_Y0*(X0_Y0+X1_Y0)
print '对于样本(%.2f, %.2f)在分类0下的概率为%f.' % (t0, t1, result0)


# 计算在分类1下对于x0和x1的概率
X0_Y1 = normal_distribution(40, X_Y1[:, 0].mean(), X_Y1[:, 0].var())
X1_Y1 = normal_distribution(20, X_Y1[:, 1].mean(), X_Y1[:, 1].var())
result1 = P_Y1*(X0_Y1+X1_Y1)
print '对于样本(%.2f, %.2f)在分类1下的概率为%f.' % (t0, t1, result1)

print '将样本(%.2f, %.2f)分类到%d.' % (t0, t1, result1 > result0)


# 用得到的模型估计已有的分类样本
results = np.zeros((30, 1), dtype=np.int)

for i in range(0, TestY.size):
    X0_Y0 = normal_distribution(TestX[i, 0], X_Y0[:, 0].mean(), X_Y0[:, 0].var())
    X1_Y0 = normal_distribution(TestX[i, 1], X_Y0[:, 1].mean(), X_Y0[:, 1].var())
    result0 = P_Y0 * (X0_Y0 + X1_Y0)
    X0_Y1 = normal_distribution(TestX[i, 0], X_Y1[:, 0].mean(), X_Y1[:, 0].var())
    X1_Y1 = normal_distribution(TestX[i, 0], X_Y1[:, 1].mean(), X_Y1[:, 1].var())
    result1 = P_Y1 * (X0_Y1 + X1_Y1)
    results[i] = result1 > result0

# 计算用得到的模型得出的分类的正确率
print '用得到的模型得出的分类正确率为: %.2f%%' % (float(sum(results == TestY)*100)/TestY.size)