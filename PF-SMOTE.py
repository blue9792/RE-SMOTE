import math
import time
import numpy as np
import random
import matplotlib.pyplot as plt
from numpy import mean
from sklearn import svm
from sklearn.model_selection import cross_val_score, RepeatedStratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier


class SMOTE(object):
    # 初始化函数
    def __init__(self, sample, sample_data, k=2, gen_num=3):
        # 需要被扩充的样本
        self.sample = sample
        # 总的样本集
        self.sample_data = sample_data
        # 获取输入数据的形状 如果输入的样本形状是10*2 意味着输入了10个样本 每个样本由2个特征构成
        self.sample_num, self.feature_len = len(self.sample), len(self.sample[0]) - 1
        # 邻近点 如果被扩充的数据有10个样本 则每个样本点最多有9个相邻的点 防止k值过大
        self.k = min(k, self.sample_num - 1)
        # 需要生成的样本的数量
        self.gen_num = gen_num
        # 定义一个数组存储生成的样本 全0数组存储生成的数据
        self.syn_data = np.zeros((self.gen_num, self.feature_len))
        # 定义一个数组存储每一个点和其临近点的坐标
        self.k_neighbor = np.zeros((self.sample_num, len(self.sample_data) - 1), dtype=int)
        # 边界少数样本索引 安全少数样本
        self.border_sample, self.safe_sample = [], [],
        # 计算少数类边界样本均值
        self.x_mean = np.zeros((1, self.feature_len))

    # 获取相邻点的函数
    def get_neighbor_point(self):
        for index, single_signal in enumerate(self.sample):
            # 获取欧式距离
            Enclidean_distance = np.array(
                [np.sum(np.square(np.array(single_signal[:self.feature_len]) - np.array(i[:self.feature_len]))) for i in
                 self.sample_data])
            # 获取欧式距离从小到大的索引排序序列
            Enclidean_distance_index = Enclidean_distance.argsort()
            # 截取k个距离最近的一个样本点的索引值
            self.k_neighbor[index] = Enclidean_distance_index[1:]

    # 分成边界少数和安全少数样本
    def border_safe(self):
        self.get_neighbor_point()
        b = np.zeros((len(self.sample), self.feature_len))
        for i in range(len(self.k_neighbor)):
            # 该样本点为少数类样本，它的最近邻为少数类样本，则该样本点为安全少数样本。如果该样本点为少数类样本，它的最近邻为多数类样本，则该样本点为边界少数样本
            if self.sample[i][-1] != self.sample_data[self.k_neighbor[i][0]][-1]:
                # 把边界少数的索引添加到border_sample中
                self.border_sample.append(i)

                b[i] = self.sample[i][:self.feature_len]
            else:
                # 把安全少数的索引添加到safe_sample中
                self.safe_sample.append(i)
        self.x_mean = np.mean(b, axis=0)

    # 合成安全少数样本
    def get_syn_safe_data(self):
        # 计算合成安全少数样本的数量
        safe_gen_num = int(len(self.safe_sample) / len(self.sample) * self.gen_num)
        gap = random.uniform(0, 1)
        # 合成的新样本
        syn_safe_data = np.zeros((safe_gen_num, self.feature_len))
        for j in range(safe_gen_num):
            # 从安全区域中随机选择一个样本点
            x = self.safe_sample[random.randint(0, len(self.safe_sample) - 1)]
            # 该样本点的近邻样本点索引 直到 最近多数类为止  不包含最近多数类
            arr = []
            for i in range(len(self.k_neighbor[0])):
                if self.sample_data[self.k_neighbor[x][i]][-1] == 1:
                    # 该少数类样本的近邻少数类样本点的索引
                    arr.append(self.k_neighbor[x][i])
                else:
                    break
            # x_safe为索引值 随机选择 arr中任一 样本点 进行新样本合成
            x_safe = arr[random.randint(0, len(arr) - 1)]
            syn_safe_data[j] = np.array(self.sample[x][:self.feature_len]) + gap * (
                    np.array(self.sample_data[x_safe][:self.feature_len]) - np.array(
                self.sample[x][:self.feature_len]))
        return syn_safe_data

    # 合成边界少数样本
    def get_syn_border_data(self):
        gap = random.uniform(0, 1)
        # x求均值 计算每一列的均值
        dist = 0.0
        # 计算欧式距离 (x_i - x_mean)^2
        for x in self.border_sample:
            dist += np.sum(np.square(self.sample[x][:self.feature_len] - self.x_mean))
        # 计算 |TS+| 先 计算模 但是因为最后一列都为0 计算的时候加入了 于是 平方 减去后 再开根号
        TS_add = np.sqrt(np.linalg.norm(self.border_sample) ** 2 - len(self.border_sample))
        # variance 方差
        variance = dist / (TS_add - 1)
        # 合成的新样本
        syn_border_data = np.zeros((len(self.border_sample), self.feature_len))
        # 对每个边界样本的索引进行依次遍历 进行新样本的合成
        j = 0
        for x in self.border_sample:
            # 找到该边界样本点的最近多数样本点
            for i in range(len(self.k_neighbor[0])):
                if self.sample_data[self.k_neighbor[x][i]][-1] == 0:
                    # 该少数类样本的最近多数类样本点的索引
                    x_nn = self.k_neighbor[x][i]
                    break
            # X-N(期望值，方差) expectations：期望值
            expectations = np.array(self.sample[x][:self.feature_len]) + gap * (
                    np.array(self.sample_data[x_nn][:self.feature_len]) - np.array(
                self.sample[x][:self.feature_len]))
            # 新样本合成
            syn_border_data[j] = np.array(1 / (np.sqrt(2 * math.pi) * np.sqrt(variance)) * np.exp(
                (-1) * ((np.array(self.sample[x][:self.feature_len]) - expectations) ** 2) / (2 * variance)))
            j += 1
        return syn_border_data


if __name__ == '__main__':
    start = time.perf_counter()
    file = open('dataset/standard/hepatitis/hepatitis-5-1.dat', 'r')
    # 总数居集 多数类数据集 少数类数据集
    data, data_negative, data_positive = [], [], []
    i = 0
    # 将少数类数据和多数类数据分开
    for line in file.read().splitlines():
        if line[0] != '@':
            arr = list(line.split(','))
            if '1' in arr[-1]:
                # 1 为positive 少数类
                arr = [float(arr[i]) for i in range(len(arr) - 1)]
                arr.append(1)
                data_positive.append(arr)
            else:
                # 0 为 negative
                arr = [float(arr[i]) for i in range(len(arr) - 1)]
                arr.append(0)
                data_negative.append(arr)
            data.append(arr)
    print("positive:" + str(len(data_positive)))
    print("negative:" + str(len(data_negative)))
    print("总：" + str(len(data)))
    print("%.3f" % (len(data_negative) / len(data_positive)))
    # 绘制多数类数据
    for i in data_negative:
        plt.scatter(i[0], i[1], c='r')
    # 绘制少数类数据
    for i in data_positive:
        plt.scatter(i[0], i[1], c='b')
    plt.show()
    # 生成对象k=3 gen_num=
    Syntheic_sample = SMOTE(data_positive, data, 3, abs(len(data_positive) - len(data_negative)))
    # 分成安全少数样本和边界少数样本
    Syntheic_sample.border_safe()
    # 合成安全少数样本
    safe_data = Syntheic_sample.get_syn_safe_data()

    # 合成边界少数样本
    border_data = Syntheic_sample.get_syn_border_data()
    # 绘制多数类数据
    for i in data_negative:
        plt.scatter(i[0], i[1], c='r')
    # 绘制少数类数据
    for i in data_positive:
        plt.scatter(i[0], i[1], c='b')
    # 绘制生成数据
    for i in safe_data:
        plt.scatter(i[0], i[1], c='g')
    for i in border_data:
        plt.scatter(i[0], i[1], c='m')
    plt.show()
    # 给生成的安全少数样本添加少数类标签
    safe_data = np.c_[safe_data, np.array([1 for _ in range(len(safe_data))])]
    # 给生成的边界少数样本添加少数类标签
    border_data = np.c_[border_data, np.array([1 for _ in range(len(border_data))])]
    # 把生成的安全少数样本添加到总样本集中
    data = np.concatenate((data, safe_data), axis=0)
    # 把生成的边界少数样本添加到总样本集中
    data = np.concatenate((data, border_data), axis=0)
    end = time.perf_counter()
    print("time:%.4f" % (end - start))
    # 样本集中最后一列标签列
    y = data[:, -1]
    # 删除标签
    X = np.delete(data, -1, axis=1)
    model = DecisionTreeClassifier(class_weight='balanced')
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=10, random_state=1)
    print("DT")
    print("auc: %.4f " % mean(cross_val_score(model, X, y, scoring='roc_auc', cv=cv)),
          "recall:%.4f " % mean(cross_val_score(model, X, y, scoring='recall', cv=cv)),
          "precision:%.4f " % mean(cross_val_score(model, X, y, scoring='precision', cv=cv)),
          "accuracy: %.4f " % mean(cross_val_score(model, X, y, scoring='accuracy', cv=cv)),
          "f1: %.4f " % mean(cross_val_score(model, X, y, scoring='f1', cv=cv)))

    knn = KNeighborsClassifier(3)  # knn模型，这里一个超参数可以做预测，当多个超参数时需要使用另一种方法GridSearchCV
    print("KNN")
    print("auc: %.4f " % mean(cross_val_score(knn, X, y, scoring='roc_auc', cv=cv)),
          "recall:%.4f " % mean(cross_val_score(knn, X, y, scoring='recall', cv=cv)),
          "precision:%.4f " % mean(cross_val_score(knn, X, y, scoring='precision', cv=cv)),
          "accuracy: %.4f " % mean(cross_val_score(knn, X, y, scoring='accuracy', cv=cv)),
          "f1: %.4f " % mean(cross_val_score(knn, X, y, scoring='f1', cv=cv)))

    SVC = svm.SVC()
    print("SVM")
    print("auc: %.4f " % mean(cross_val_score(SVC, X, y, scoring='roc_auc', cv=cv)),
          "recall:%.4f " % mean(cross_val_score(SVC, X, y, scoring='recall', cv=cv)),
          "precision:%.4f " % mean(cross_val_score(SVC, X, y, scoring='precision', cv=cv)),
          "accuracy: %.4f " % mean(cross_val_score(SVC, X, y, scoring='accuracy', cv=cv)),
          "f1: %.4f " % mean(cross_val_score(SVC, X, y, scoring='f1', cv=cv)))