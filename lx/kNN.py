import numpy as np
import pandas as pd
from tqdm import tqdm

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

def createDataSet():
    group = np.array([[1, 1.1], [1, 1], [0, 0], [0, 0.1]])
    labels = ['A', 'A', 'B', 'B']
    return group, labels

def readDataSet():
    group = pd.read_csv("data_proc.txt")
    dataSet = group[['item_name','height','weight','rating']]
    labels = group['fit']
    dataSet = np.array(dataSet)
    labels = np.array(labels)
    return dataSet, labels

if __name__ == '__main__':
    # dataSet, labels = createDataSet()
    # r = classify_two([0, 0.2], dataSet, labels, 3)
    # r = classify([0, 0.2], dataSet, labels, 3)
    # print(r)
    epoch = 50000
    count = 0
    dataSet, labels = readDataSet()
    for i in tqdm(range(epoch)):
        r = classify(dataSet[i], dataSet, labels, 10)
        if(r == labels[i]):
            count += 1
    print("ACC: ", count/epoch*100)