# 定义IPF函数，输入为过采样后的数据集X_samp,y_samp，输出为经过清洗后的数据集X_clean,y_clean
import numpy as np
from sklearn.ensemble import RandomForestClassifier


def IPF(X_samp, y_samp):
    # 初始化清洗后的数据集为空列表
    X_clean = []
    y_clean = []

    # 初始化随机森林分类器，并用过采样后的数据集训练它
    rf = RandomForestClassifier()
    rf.fit(X_samp, y_samp)

    # 对每个训练样本进行预测，并计算其概率分布（即被判定为每个类别的概率）
    y_pred = rf.predict(X_samp)
    y_prob = rf.predict_proba(X_samp)
    print("y_pred")
    print(y_pred)
    print("y_prob")
    print(y_prob)
    # 对每个训练样本，计算其与同类样本的平均距离和与异类样本的平均距离
    d_intra = np.zeros(len(X_samp))
    d_inter = np.zeros(len(X_samp))
    for i in range(len(X_samp)):
        same_class_index = np.where(y_samp == y_samp[i])[0]
        diff_class_index = np.where(y_samp != y_samp[i])[0]
        d_intra[i] = np.mean(np.linalg.norm(X_samp[i] - X_samp[same_class_index], axis=1))
        d_inter[i] = np.mean(np.linalg.norm(X_samp[i] - X_samp[diff_class_index], axis=1))

    # 对每个训练样本，根据其预测结果、概率分布和距离指标，判断是否为噪声或边界样本，并决定是否保留
    for i in range(len(X_samp)):
        # 如果预测结果与真实标签一致，并且预测概率大于等于0.7，则保留该样本
        print(y_pred[i], y_samp[i], y_prob[i][y_pred[i]])
        if y_pred[i] == y_samp[i] and y_prob[i][y_pred[i]] >= 0.7:
            X_clean.append(X_samp[i])
            y_clean.append(y_samp[i])
        # 如果预测结果与真实标签不一致，并且预测概率小于等于0.3，则认为该样本是噪声，删除该样本
        elif y_pred[i] != y_samp[i] and y_prob[i][y_pred[i]] <= 0.3:
            continue
        # 如果预测结果与真实标签不一致，但是预测概率大于0.3，则认为该样本是边界样本，根据其距离指标判断是否保留
        else:
            # 如果同类平均距离小于异类平均距离，则保留该样本
            if d_intra[i] < d_inter[i]:
                X_clean.append(X_samp[i])
                y_clean.append(y_samp[i])
            # 否则删除该样本
            else:
                continue

    # 将清洗后的数据集转换为数组，并返回
    X_clean = np.array(X_clean)
    y_clean = np.array(y_clean)
    return X_clean, y_clean


from sklearn.datasets import make_classification

# 生成一个不平衡的二分类数据集
X, y = make_classification(n_samples=1000, n_features=10, n_classes=2,
                           weights=[0.9, 0.1], random_state=1)
print(X)
print(y)
# 查看原始数据集的类别分布
print('Original dataset:')
print('Minority class:', np.sum(y == 1))
print('Majority class:', np.sum(y == 0))

# 调用SMOTE-IPF函数，得到过采样和清洗后的数据集
X_final, y_final = IPF(X, y)

# 查看最终数据集的类别分布
print('Final dataset:')
print('Minority class:', np.sum(y_final == 1))
print('Majority class:', np.sum(y_final == 0))
