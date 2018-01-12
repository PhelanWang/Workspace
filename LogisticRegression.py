# coding: utf-8


import numpy as np
import matplotlib.pyplot as plt


# 读取数据到data
data = np.loadtxt('./data/logistic_data.txt', delimiter=',')

X = data[:, 0:2]
Y = data[:, 2:3]
# m=样本数
m = X.shape[0]
# X加上一列1
X = np.hstack((np.ones((m, 1), dtype=np.float), X))
# 分别找出类别为1和为0的行数
index_one = np.where(data[:, 2:3] == 1)[0]
index_zero = np.where(data[:, 2:3] == 0)[0]

data_one = data[index_one, :]
data_zero = data[index_zero, :]

# 画出类别的散点图
plt.scatter(data_one[:, 0:1], data_one[:, 1:2], marker='x', linewidth=0.5)
plt.scatter(data_zero[:, 0:1], data_zero[:, 1:2], marker='o', linewidth=0.5)


# 逻辑函数
def sigmoid(z):
    return np.longfloat(1.0/(1.0+np.exp(-z)))


# 计算代价
def compute_cost(X, Y, theta, m, lambda_param):
    # h = 1/(1+e^(theta'*X'))
    # J = -(log(h)*(-Y)+log(1-h)*(1-Y))/m
    h = sigmoid(theta.transpose().dot(X.transpose()))
    print h.shape
    theta_temp = theta.copy()
    theta_temp[0] = 0
    theta_temp = lambda_param/(2*m)*theta_temp.transpose().dot(theta_temp)
    J = (np.log(h).dot(-Y)-np.log(1-h).dot(1-Y))/m + theta_temp
    return J


# 梯度下降算法
def gradient_descent(X, Y, theta, alpha, lambda_param, m, count):
    for i in range(0, count):
        theta_temp = theta.copy()
        theta_temp[0] = 0
        theta_temp = lambda_param/m*theta_temp

        h = sigmoid(theta.transpose().dot(X.transpose()))
        theta = theta - (alpha*(X.transpose().dot(h.transpose()-Y)/m)) - theta_temp
    return theta


# 初始化
alpha = 0.001
lambda_param = 3

theta = np.array([[-21], [1], [1]])
J = compute_cost(X, Y, theta, m, lambda_param)
print J.flat.next()


# 运行梯度下降算法求解theta
theta = gradient_descent(X, Y, theta, alpha, lambda_param, m, 1000)

print theta
J = compute_cost(X, Y, theta, m, lambda_param)
print J.flat.next()

x1_min = np.min(X[:, 1:2])-2
x1_max = np.max(X[:, 1:2])-2

x2_min = -(1/theta[2])*(theta[1]*x1_min+theta[0])
x2_max = -(1/theta[2])*(theta[1]*x1_max+theta[0])

plt.plot([x1_min, x1_max], [x2_min, x2_max], linewidth=0.5)

test_x1 = 90
test_x2 = 80
predict = sigmoid(np.array([1, test_x1, test_x2]).dot(theta))
plt.scatter(test_x1, test_x2, marker='+', color='r', linewidth=0.5)
print 'predict:', predict >= 0.5


plt.show()