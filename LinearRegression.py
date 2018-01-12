# coding:utf-8

import numpy as np
import matplotlib.pyplot as plt
"""
线性回归算法
数据来自data/data1.text,数据有两列,第一列是X,第二列是Y
分别用梯度下降算法和最小二乘法求解theta,并输出theta,代价,绘制出拟合直线
用最小二乘法求得的theta预测
"""


# 计算代价
def compute_cost(X, Y, theta, m):
    # 计算代价sum((X*theta - Y)^2)/2*m
    return ((X.dot(theta) - Y) ** 2).sum() / (2 * m)


# 梯度下降算法,求解theta, count是迭代次数
def gradient_descent(X, Y, theta, alpha, count):
    # 梯度下降算法求解 alpha*sum(X'(X*theta - Y))/m
    for i in range(0, count):
        theta = theta - alpha / m * (X.transpose().dot((X.dot(theta) - Y)))
    return theta


# 最小二乘法求解theta
def least_square_method(X, Y, theta, m):
    # 最小二乘法求解
    return np.linalg.inv((X.transpose().dot(X))).dot(X.transpose()).dot(Y)


# 读取数据到data
data = np.loadtxt('./data/linear_data.txt', delimiter=',')
X = data[:, 0]
Y = data[:, 1]
X.shape = X.size, 1
Y.shape = Y.size, 1
# 样本数量
m = X.shape[0]
X_temp = X
min_x = X[:, 0].min()
max_x = X[:, 0].max()
# 画出数据的散点图
plt.scatter(X, Y, marker='x')
# 给X添加第一列1
X = np.hstack((np.ones((X.size, 1), dtype=np.float), X))
# 初始化theta参数
theta = np.ones((X.shape[1], 1))


# 输出初始化的theta,代价,绘制拟合直线
print theta
print compute_cost(X, Y, theta, m)
# 绘制拟合的直线
plt.plot(np.array([min_x, max_x]), np.array([theta[0] + min_x * theta[1], theta[0] + max_x * theta[1]]), color='r', label='init')


# 初始化学习速率
alpha = 0.01


# 梯度下降算法求解 alpha*sum(X'(X*theta - Y))/m
# 输出梯度下降算法求解的theta,代价,绘制拟合直线
g_theta = gradient_descent(X, Y, theta, alpha, 1000)
print g_theta
print compute_cost(X, Y, g_theta, m)
# 绘制拟合的直线
plt.plot(np.array([min_x, max_x]), np.array([g_theta[0] + min_x * g_theta[1], g_theta[0] + max_x * g_theta[1]]), color='b', label='gradient')


# 最小二乘法求解
l_theta = least_square_method(X, Y, theta, m)
print l_theta
print compute_cost(X, Y, l_theta, m)
# 绘制拟合的直线
plt.plot(np.array([min_x, max_x]), np.array([l_theta[0] + min_x * l_theta[1], l_theta[0] + max_x * l_theta[1]]), color='y', label='least')


# 预测当x=22时的y的值
x = 11
y = np.array([1, x]).dot(l_theta)
plt.scatter(x, y, marker='x', color='r')
print 'x = ', x, '时，预测值为：', y

plt.legend()
plt.show()

