import numpy as np
from matplotlib import pyplot as plt
from rpy2 import robjects


class SMOTE(object):
    def __init__(self, sample, sample_data, k=6, gen_num=3):
        # 需要被扩充的样本 为少数类样本
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
    # y:少数类的数量
    def IPF(self, after_data):
        robjects.r('''
        f<-function(D){D<-read}
        # pi*r
            # f<-function(D,y){
            #         library(RWeka)
            #         library(DMwR)
            #         library(data.table)
            # 
            #         args = commandArgs(trailingOnly=TRUE)
            # 
            # 
            #         # print("Number of sub-datsets to make : ")
            #         n=9 # as.integer(readLines("stdin",n=1))
            # 
            #         print("Number of iterations to identify noisy examples : ")
            #         k=3 # as.integer(readLines("stdin",n=1))
            # 
            #         # print("1 for majority, 0 for consensus : ")
            #         voting=1 # as.integer(readLines("stdin",n=1))
            # 
            #         # print("Percentage of original dataset considered for filtering : ")
            #         p=1.0 # as.numeric(readLines("stdin",n=1))
            # 
            #         # print("Specify k as in knn of SMOTE : ")
            #         k_knn=11 # k_knn=as.integer(readLines("stdin",n=1))
            # 
            # 
            # 
            #         # shuffleSmotedData <- function(dataset)
            #         # {
            #         # 	for (i in 1:3)
            #         # 		dataset = dataset[sample(nrow(dataset)),]
            #         # 	return (dataset)
            #         # }
            # 
            #         P = sapply((length(d[,1])*p)/100,as.integer)	#maximum no of tolerable noisy instances in each iteration
            # 
            #         i_k=0
            #         while(i_k<k)
            #         {
            #             L=length(D[,1])	#no of rows in D(Smoted Dataset)
            #             step = as.integer(L/n)	#length of each sub-dataset
            # 
            #             models = vector(mode="list",length=n)
            #             # D=shuffleSmotedData(D)	#shuffle D randomly such that each sub-datset contains roughly equal 'Y' and 'N'
            # 
            #             for (i in 0:(n-2))
            #             {
            #                 models[[i+1]] = J48(as.factor(Class)~.,D[(i*step+1):((i+1)*step),])	#append each tree generated through subsets into models
            #             }
            #             models[[n]] = J48(as.factor(Class)~.,D[(n-1)*step+1:length(D[,1]),])
            #             predictions = vector(mode="list",length=n)
            #             for (i in 1:n)
            #             {
            #                 predictions[[i]] = predict(models[[i]],newdata=D)	#run the whole dataset through each model
            #             }
            # 
            #             temp_indices=c()	#stores indices of noisy data
            #             i_t=0	#counter for above variable
            #             if (voting==1)	#for majority
            #             {
            #                 for (i in 1:L)
            #                 {
            #                     positive=0
            #                     negetive=0
            #                     for (j in 1:n)
            #                     {
            #                         if (D$Class[i]==predictions[[j]][i])
            #                             positive=positive+1
            #                         else
            #                             negetive=negetive+1
            #                     }
            #                     if (positive<negetive)
            #                     {
            #                         i_t=i_t+1
            #                         temp_indices[i_t]=i
            #                     }
            #                 }
            #             }
            #             else 	#for consensus
            #             {
            #                 for (i in 1:L)
            #                 {
            #                     misclassified_by_all=1
            #                     for (j in 1:n)
            #                     {
            #                         if (D$Class[i]==predictions[[j]][i])
            #                         {
            #                             misclassified_by_all=0
            #                             break
            #                         }
            #                     }
            #                     if (misclassified_by_all==1)
            #                     {
            #                         i_t=i_t+1
            #                         temp_indices[i_t]=i
            #                     }
            #                 }
            #             }
            #             D=D[-temp_indices,]	#remove noisy examples
            # 
            #             if (i_t<=P)	#no of noisy examples less than P : increment counter
            #                 i_k=i_k+1
            #             else 	#set counter to 0
            #                 i_k=0
            #             print(".")
            #         }
            # 
            #         print("Filtered Data")
            #         print(table(D$Class))
            #         print(D)
            #         fwrite(D,args[2])	#write final dataset to another csv
            #         }        
            
                ''')
        print(robjects.r['f'](after_data))


if __name__ == '__main__':
    file = open('dataset/SMOTE/ecoli1/result.tst', 'r')

    # 总数居集 多数类数据集 少数类数据集
    data, data_negative, data_positive = [], [], []
    i = 0
    # 将少数类数据和多数类数据分开
    for line in file.read().splitlines():
        if line[0] != '@':
            arr = list(line.split(','))
            if 'positive' in arr[-1]:
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
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签

    # # 绘制多数类数据
    # for i in data_negative:
    #     plt.scatter(i[0], i[1], c='r')
    # plt.scatter(i[0], i[1], c='r', label="majority class")
    # # 绘制少数类数据
    # for i in data_positive:
    #     plt.scatter(i[0], i[1], c='b')
    # plt.scatter(i[0], i[1], c='b', label="minority class")
    # plt.legend()
    # plt.show()
    # 生成对象k=3 gen_num=
    Syntheic_sample = SMOTE(data_positive, data, 3, abs(len(data_positive) - len(data_negative)))
    # 分成安全少数样本和边界少数样本
    y = len(data_positive)
    # 不平衡率
    # IR = N- / N+
    IR = len(data_negative) / len(data_positive)
    print("IR:" + str(IR))
    print(data)
    Syntheic_sample.IPF(data)
