import numpy as np
import pandas as pd
from tqdm import tqdm

def attempt(epoch):
    # 对处理过的数据集中所有数据预测为1（Ture to Size)，正确率都可以达到73.06%
    group = pd.read_csv("data_proc.txt")
    dataSet = group[['item_name','height','weight','rating']]
    labels = group['fit']
    acc =  0
    for i in range(epoch):
        if labels[i] == 1:
            acc += 1
    print(acc/epoch*100,"%")

def classify(inX, dataSet, labels, k):
    """
    定义knn算法分类器函数
    :param inX: 测试数据
    :param dataSet: 训练数据
    :param labels: 分类类别
    :param k: k值
    :return: 所属分类
    """

    dataSetSize = dataSet.shape[0]  #shape（m, n）m列n个特征
    diffMat = np.tile(inX, (dataSetSize, 1)) - dataSet
    sqDiffMat = diffMat ** 2
    sqDistances = sqDiffMat.sum(axis=1)
    distances = sqDistances ** 0.5  #欧式距离
    sortedDistIndicies = distances.argsort()  #排序并返回index

    classCount = {}
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1 #default 0

    sortedClassCount = sorted(classCount.items(), key=lambda d:d[1], reverse=True)
    return sortedClassCount[0][0]

def classify_two(inX, dataSet, labels, k):
    m, n = dataSet.shape   # shape（m, n）m列n个特征
    # 计算测试数据到每个点的欧式距离
    distances = []
    for i in range(m):
        sum = 0
        for j in range(n):
            sum += (inX[j] - dataSet[i][j]) ** 2
        distances.append(sum ** 0.5)

    sortDist = sorted(distances)

    # k 个最近的值所属的类别
    classCount = {}
    for i in range(k):
        voteLabel = labels[ distances.index(sortDist[i])]
        classCount[voteLabel] = classCount.get(voteLabel, 0) + 1 # 0:map default
    sortedClass = sorted(classCount.items(), key=lambda d:d[1], reverse=True)
    return sortedClass[0][0]

def readDataSet():
    group = pd.read_csv("data_proc.txt")
    dataSet = group[['item_name','height','weight','rating']]
    labels = group['fit']
    dataSet = np.array(dataSet)
    labels = np.array(labels)
    return dataSet, labels

def TrainSet():
    group0 = pd.read_csv("data_fit_0.txt")
    group1 = pd.read_csv("data_fit_1.txt")
    group2 = pd.read_csv("data_fit_2.txt")
    dataSet0 = np.array(group0[['item_name','height','weight','rating']])
    dataSet1 = np.array(group1[['item_name','height','weight','rating']])
    dataSet2 = np.array(group2[['item_name','height','weight','rating']])
    labels0 = np.array(group0['fit'])
    labels1 = np.array(group1['fit'])
    labels2 = np.array(group2['fit'])
    dataSet = np.concatenate((dataSet0[:2500, :], dataSet1[:2500, :], dataSet2[:2500, :]), axis=0)
    labels = np.concatenate((labels0[:2500], labels1[:2500], labels2[:2500]), axis=0)
    # print(dataSet.size)
    # print(labels.size)
    return dataSet, labels

def TestSet():
    group0 = pd.read_csv("data_fit_0.txt")
    group1 = pd.read_csv("data_fit_1.txt")
    group2 = pd.read_csv("data_fit_2.txt")
    dataSet0 = np.array(group0[['item_name','height','weight','rating']])
    dataSet1 = np.array(group1[['item_name','height','weight','rating']])
    dataSet2 = np.array(group2[['item_name','height','weight','rating']])
    labels0 = np.array(group0['fit'])
    labels1 = np.array(group1['fit'])
    labels2 = np.array(group2['fit'])
    dataSet = np.concatenate((dataSet0[2500:, :], dataSet1[2500:, :], dataSet2[2500:, :]), axis=0)
    labels = np.concatenate((labels0[2500:], labels1[2500:], labels2[2500:]), axis=0)
    # print(dataSet.size)
    # print(labels.size)
    return dataSet, labels


if __name__ == '__main__':
    # 直接对空缺赋所有数的平均值162and140  epoch=500 acc=79%，epoch=1000 acc=78.2%，epoch=2000 acc=77.65%，epoch=20000 acc=76.795
    # 直接对空缺赋所有数的平均值164and143  epoch=500 acc=78.8%，epoch=1000 acc=78.4%，epoch=2000 acc=78.05%
    # 直接对空缺赋所有数的平均值164and141  epoch=500 acc=79.2%，epoch=1000 acc=78.7%，epoch=2000 acc=78.15%
    # 直接对空缺赋所有数的平均值164and140  epoch=500 acc=79.4%，epoch=1000 acc=78.8%，epoch=2000 acc=78.2%
    # 直接对空缺赋所有数的平均值166and145  epoch=500 acc=78.2%，epoch=1000 acc=77.9%，epoch=2000 acc=77.8%
    # 对不同类型空缺赋不同的平均值 epoch=500 acc=78%，epoch=1000 acc=77.4%，epoch=2000 acc=77.25%，epoch=20000 acc=76.305
    count = 0
    dataSet, labels = TrainSet()
    dataSet_test, labels_test = TestSet()
    epoch = labels_test.size
    for i in tqdm(range(epoch)):
        r = classify(dataSet_test[i], dataSet[:], labels, 20)
        if(r == labels_test[i]):
            count += 1

    print("ACC: ", count/(epoch)*100)
