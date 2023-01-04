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
    # group0 = pd.read_csv("data_fit_0.txt")
    # group1 = pd.read_csv("data_fit_1.txt")
    # group2 = pd.read_csv("data_fit_2.txt")
    # dataSet0 = np.array(group0[['item_name','height','weight','rating']])
    # dataSet1 = np.array(group1[['item_name','height','weight','rating']])
    # dataSet2 = np.array(group2[['item_name','height','weight','rating']])
    # labels0 = np.array(group0['fit'])
    # labels1 = np.array(group1['fit'])
    # labels2 = np.array(group2['fit'])
    # dataSet = np.concatenate((dataSet0[:2500, :], dataSet1[:2500, :], dataSet2[:2500, :]), axis=0)
    # labels = np.concatenate((labels0[:2500], labels1[:2500], labels2[:2500]), axis=0)
    # print(dataSet.size)
    # print(labels.size)
    labels = np.load('label2vector.npy')
    dataSet = np.load('data2vector.npy')
    return dataSet, labels

def TestSet():
    # group0 = pd.read_csv("data_fit_0.txt")
    # group1 = pd.read_csv("data_fit_1.txt")
    # group2 = pd.read_csv("data_fit_2.txt")
    # dataSet0 = np.array(group0[['item_name','height','weight','rating']])
    # dataSet1 = np.array(group1[['item_name','height','weight','rating']])
    # dataSet2 = np.array(group2[['item_name','height','weight','rating']])
    # labels0 = np.array(group0['fit'])
    # labels1 = np.array(group1['fit'])
    # labels2 = np.array(group2['fit'])
    # dataSet = np.concatenate((dataSet0[2500:, :], dataSet1[2500:, :], dataSet2[2500:, :]), axis=0)
    # labels = np.concatenate((labels0[2500:], labels1[2500:], labels2[2500:]), axis=0)
    # print(dataSet.size)
    # print(labels.size)
    labels = np.load('label2vector.npy')
    dataSet = np.load('data2vector.npy')
    return dataSet, labels


if __name__ == '__main__':
    '''
    epoch = 500, acc = 81.2
    epoch = 2000, acc = 81.9
    '''
    
    dataSet, labels = TrainSet()
    dataSet_test, labels_test = TestSet()
    m, n = dataSet.shape
    # count = 0
    # epoch = 2000
    # for i in tqdm(range(epoch)):
    #     r = classify(dataSet_test[i], dataSet[:], labels, 5)
    #     if(r == labels_test[i]):
    #         count += 1

    # print("ACC: ", count/(epoch)*100)


    Store_data = dataSet[0:20]
    Store_label = labels[0:20]
    # print(Store_data)
    # print(dataSet[0].reshape(1,12).shape)
    # print(Store_label[:].shape)
    # print(labels[0].reshape(1,1))
    # acc = []
    # for t in range(3,10):
    #     for s in range(3,10):
    #         for i in tqdm(range(m)):
    #             r = classify(dataSet[i], Store_data[:], Store_label[:], t)
    #             if r != labels[i]:
    #                 Store_data = np.concatenate((Store_data, dataSet[i].reshape(1,12)), axis=0)
    #                 Store_label = np.append(Store_label, labels[i])
    #         epoch = 22071
    #         total = 0
    #         count = 0
    #         for i in tqdm(range(epoch)):
    #             if labels_test[i] == -1:
    #                 total += 1
    #                 r = classify(dataSet_test[i], Store_data[:], Store_label[:], s)
    #                 if(r == labels_test[i]):
    #                     count += 1
    #         acc.append(count/total*100)
    # print(acc)
    # print(acc.index(max(acc)))
    # np.save('acc.npy',np.array(acc))

    # （9，9）做到对三个类别分类准确率分别为34.23%，80.21%，34.14%
    # （10，5）做到对三个类型分类准确率分别为51.69%，70.11%，50.76%
    count = 0
    for i in tqdm(range(m)):
        r = classify(dataSet[i], Store_data[:], Store_label[:], 10)
        if r != labels[i]:
            Store_data = np.concatenate((Store_data, dataSet[i].reshape(1,12)), axis=0)
            Store_label = np.append(Store_label, labels[i])
    epoch = 22071
    total = 0
    count = 0
    for i in tqdm(range(epoch)):
        if labels_test[i] == 0:
            total += 1
            r = classify(dataSet_test[i], Store_data[:], Store_label[:], 5)
            if(r == labels_test[i]):
                count += 1
    print("ACC: ", count/(total)*100)