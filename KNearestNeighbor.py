# coding: utf-8
import numpy as np
import matplotlib.pyplot as plt


def euclidean_distance(v1, v2):
    return np.sqrt(np.sum((v1 - v2)**2))


# 返回样本的分类
def get_category(X, Y, x, k):
    distance = np.zeros((Y.size, 1), dtype=np.float)
    # 计算新样本到其他样本间的距离
    for i in range(0, Y.size):
        distance[i, 0] = euclidean_distance(X[i, :], x)
    # 将得到的结果排序
    sort_data = np.hstack((distance, data[:70, :]))
    sort_data = sort_data[sort_data[:, 0].argsort()]
    # 选取前k个
    result_data = sort_data[0:k, :]
    count_c0 = np.sum(result_data[:, 3] == 0)
    count_c1 = np.sum(result_data[:, 3] == 1)
    return count_c1 > count_c0

data = np.loadtxt('data/logistic_data.txt', delimiter=',')


X = data[:70, 0:2]
Y = data[:70, 2:3]


TestX = data[70:, 0:2]
TestY = data[70:, 2:3]

# 找出分类为0和分类为1的样本
X_C0 = X[np.where(Y == 0)[0], 0:2]
X_C1 = X[np.where(Y == 1)[0], 0:2]

plt.scatter(X_C0[:, 0], X_C0[:, 1], marker='+', linewidths=0.1)
plt.scatter(X_C1[:, 0], X_C1[:, 1], marker='x', linewidths=0.1)


input_x = raw_input("请输入一个点: ")
input_x = input_x.split(' ')

new_x = [int(input_x[0]), int(input_x[1])]
k = 9

distance = np.zeros((Y.size, 1), dtype=np.float)

# 计算新样本到其他样本间的距离
for i in range(0, Y.size):
    distance[i, 0] = euclidean_distance(X[i, :], new_x)

# 将得到的结果排序
sort_data = np.hstack((distance, data[:70, :]))
sort_data = sort_data[sort_data[:, 0].argsort()]

# 选取前k个
result_data = sort_data[0:k, :]

count_c0 = np.sum(result_data[:, 3] == 0)
count_c1 = np.sum(result_data[:, 3] == 1)

print "选取的 %d 个点中,为0的点有 %d 个,为1的点有 %d 个." % (k, count_c0, count_c1)
print "将 (%d, %d) 分类到 %d ." % (new_x[0], new_x[1], count_c1 > count_c0)


test_result = np.zeros((30, 1), dtype=np.float)
print get_category(X, Y, new_x, k)

for i in range(0, TestY.size):
    test_result[i, :] = get_category(X, Y, TestX[i, :], k)

print "测试集中分类的正确率为: %.2f." % (sum(test_result == TestY)/(float(TestY.size))*100)

plt.scatter(new_x[0], new_x[1], marker='o', linewidths=0.1)
plt.show()
