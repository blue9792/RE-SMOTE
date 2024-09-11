import math
import time
import numpy as np
import random
import matplotlib.pyplot as plt
from numpy import mean
from sklearn import svm
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score
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
        self.k_neighbor = np.zeros((self.sample_num, self.k), dtype=int)

    # 获取相邻点的函数
    def get_neighbor_point(self):
        for index, single_signal in enumerate(self.sample):
            # 获取欧式距离
            Enclidean_distance = np.array(
                [np.sum(np.square(np.array(single_signal[:self.feature_len]) - np.array(i[:self.feature_len]))) for i in
                 self.sample_data])
            # 获取欧式距离从小到大的索引排序序列
            Enclidean_distance_index = Enclidean_distance.argsort()
            # 截取k个距离最近的样本的索引值
            self.k_neighbor[index] = Enclidean_distance_index[1:self.k + 1]

    # 生成部分 每次选择一个中心样本，然后选择一个它的临近样本，生成合成样本
    def get_syn_data(self):
        for i in range(self.gen_num):
            # 随机选择的中心样本点的索引
            key = random.randint(0, self.sample_num - 1)
            # 随机选择的中心样本点的邻近样本点中的随机一个
            K_neighbor_point = self.k_neighbor[key][random.randint(0, self.k - 1)]
            # 随机选择的当前样本中前k近的样本点中的随机一个
            gap = np.array(self.sample_data[K_neighbor_point][:self.feature_len]) - np.array(
                self.sample[key][:self.feature_len])
            # 公式 生成 = 被选中作为中心的样本 + 0到1中的一个数*(被选中作为中心的样本-被选中作为中心的样本的临近样本点中的随机一个)
            self.syn_data[i] = np.array(self.sample[key][:self.feature_len]) + random.uniform(0, 1) * gap
        return self.syn_data

    # 计算enn 判断每个样本点是否为噪声样本 如果该样本点的多数邻近样本与所选样本点 不属于同一类 则为噪声样本
    def get_wenn(self, after_data, IR):
        # 噪声样本点集合
        noise = []
        # 权重系数计算
        IR_pos, IR_neg = 1 / (1 + IR), IR / (1 + IR)
        # e^((IR+)^m)
        n_pos, n_neg = math.exp(IR_pos ** self.feature_len), math.exp(IR_neg ** self.feature_len)
        for index, single_signal in enumerate(after_data):
            # 获取欧式距离
            Enclidean_distance = np.array(
                [np.sum(np.square(np.array(single_signal[:self.feature_len]) - np.array(i[:self.feature_len]))) for i in
                 after_data])
            # 权重系数 * 原距离
            for i in range(len(Enclidean_distance)):
                if after_data[i][-1] == 1:
                    Enclidean_distance[i] = Enclidean_distance[i] * n_pos
                else:
                    Enclidean_distance[i] = Enclidean_distance[i] * n_neg
            # 获取欧式距离从小到大的索引排序序列
            Enclidean_distance_index = Enclidean_distance.argsort()
            # 截取k个距离最近的样本的索引值
            k_enn = Enclidean_distance_index[1:self.k + 1]
            # 找到该样本点最近样本的类别
            class_sum = 0
            for x in k_enn:
                class_sum += after_data[x][-1]
            # 判断该样本点 所选近邻样本点的类别 此处用和代替 因为1为positive 少数类 如果最近邻多数为少数类 而自己为0 则为噪声样本
            if (class_sum / self.k > 0.5 and after_data[index][-1] == 0) or (class_sum / self.k < 0.5 and
                                                                             after_data[index][-1] == 1):
                noise.append(index)
        return noise


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
    for i in data_negative:
        plt.scatter(i[0], i[1], c='r')
    # 绘制少数类数据
    for i in data_positive:
        plt.scatter(i[0], i[1], c='b')

    plt.show()
    # 生成对象k=3 gen_num=
    Syntheic_sample = SMOTE(data_positive, data, 3, abs(len(data_positive) - len(data_negative)))
    # # 生成数据
    new_data = Syntheic_sample.get_syn_data()
    # smote合成之后
    # 绘制多数类数据
    for i in data_negative:
        plt.scatter(i[0], i[1], c='r')
    # 绘制少数类数据
    for i in data_positive:
        plt.scatter(i[0], i[1], c='b')
    # 绘制生成数据
    for i in new_data:
        plt.scatter(i[0], i[1], c='g')
    plt.show()

    # enn之后
    # 给生成的少数类数据添加少数类标签
    new_data = np.c_[new_data, np.array([1 for _ in range(len(new_data))])]
    # 把生成的少数类 合并到总数据集中
    data = np.concatenate((data, new_data), axis=0)
    # IR = N- / N+
    del_noise = Syntheic_sample.get_wenn(data, len(data_negative) / len(data_positive))
    # 去除噪声数据
    data = np.delete(data, del_noise, axis=0)
    # 绘制生成数据
    for i in data:
        if int(i[-1]) == 0:
            # 多数类
            plt.scatter(i[0], i[1], c='r')
        else:
            # 少数类
            plt.scatter(i[0], i[1], c='b')
    plt.show()
    end = time.perf_counter()
    print("time:%.4f" % (end - start))
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