import numpy as np
import pandas as pd
from tqdm import tqdm
from data_proc import *

def classify(inX, dataSet, labels, k):
    """
    定义knn算法分类器函数
    :param inX: 测试数据
    :param dataSet: 训练数据
    :param labels: 分类类别
    :param k: k值
    :return: 所属分类
    """

    dataSetSize = dataSet.shape[0]
    diffMat = np.tile(inX, (dataSetSize, 1)) - dataSet
    # sqDiffMat = diffMat ** 2
    sqDiffMat = np.abs(diffMat)
    sqDistances = sqDiffMat.sum(axis=1)
    # distances = sqDistances ** 0.5  # Euclidean distance
    distances = sqDistances
    sortedDistIndicies = distances.argsort()  # Sort and return index

    classCount = {}
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1 #default 0

    sortedClassCount = sorted(classCount.items(), key=lambda d:d[1], reverse=True)
    return sortedClassCount[0][0]

def classify_two(inX, dataSet, labels, k):
    # Like the classify function, but is non-matrix implemented and has a slower computation rate
    m, n = dataSet.shape   # shape（m, n）m列n个特征
    # Calculate the Euclidean distance from the test data to each point
    distances = []
    for i in range(m):
        sum = 0
        for j in range(n):
            sum += (inX[j] - dataSet[i][j]) ** 2
        distances.append(sum ** 0.5)

    sortDist = sorted(distances)

    # The category to which the k nearest values belong
    classCount = {}
    for i in range(k):
        voteLabel = labels[ distances.index(sortDist[i])]
        classCount[voteLabel] = classCount.get(voteLabel, 0) + 1 # 0:map default
    sortedClass = sorted(classCount.items(), key=lambda d:d[1], reverse=True)
    return sortedClass[0][0]

def TrainSet():
    train_proc('train_data_all.json')
    dataSet = data2vector('train_proc.txt')
    labels = label2vector()
    return dataSet, labels

def TestSet(data_path:str):
    test_proc(data_path)
    dataSet = data2vector('test_proc.txt')
    return dataSet

def balance_method():
    '''
    Add the processing of unbalanced data. Details are shown in report.
    '''
    dataSet, labels = TrainSet()
    m, _ = dataSet.shape
    Store_data = dataSet[0:15]
    Store_label = labels[0:15]
    
    # For processing parts with unbalanced data volume, redundant data is removed and data volume is reduced
    for i in tqdm(range(m)):
        r = classify(dataSet[i], Store_data[:], Store_label[:], 5)
        if r != labels[i]:
            Store_data = np.concatenate((Store_data, dataSet[i].reshape(1,10)), axis=0)
            Store_label = np.append(Store_label, labels[i])
    
    np.save('Store_data.npy', Store_data)       # Save the supporting datset in the file 'Store_data.npy'
    np.save('Store_label.npy', Store_label)     # Save the supporting labels in the file 'Store_label.npy'


# def bag_method():
#     labels = np.load('label2vector.npy')
#     dataSet = np.load('data2vector.npy')
#     m, _ = dataSet.shape
#     index1 = []
#     index2 = []
#     index3 = []
#     for i in range(m):
#         if labels[i] == 1:
#             index1.append(i)
#         elif labels[i] == 2:
#             index2.append(i)
#         elif labels[i] == 3:
#             index3.append(i)
#     data1 = dataSet[index1]
#     label1 = labels[index1]
#     data2 = dataSet[index2]
#     label2 = labels[index2]
#     data3 = dataSet[index3]
#     label3 = labels[index3]
#     bag1_data = np.concatenate((data1,data2[:3400],data3), axis=0)
#     bag2_data = np.concatenate((data1,data2[:6800],data3), axis=0) 
#     bag3_data = np.concatenate((data1,data2[:10200],data3), axis=0)
#     bag4_data = np.concatenate((data1,data2[:13600],data3), axis=0)
#     bag5_data = np.concatenate((data1,data2[13600:],data3), axis=0)   
#     bag1_label = np.concatenate((label1,label2[:3400],label3), axis=0)
#     bag2_label = np.concatenate((label1,label2[:6800],label3), axis=0)
#     bag3_label = np.concatenate((label1,label2[:10200],label3), axis=0)
#     bag4_label = np.concatenate((label1,label2[:13600],label3), axis=0)
#     bag5_label = np.concatenate((label1,label2[13600:],label3), axis=0)
#     epoch = 22071
#     count = 0
#     classCount = {}
#     dataSet_test, labels_test = TestSet()
#     for i in tqdm(range(epoch)):
#         r1 = classify(dataSet_test[i], bag1_data[:], bag1_label, 7)
#         r2 = classify(dataSet_test[i], bag2_data[:], bag2_label, 7)
#         r3 = classify(dataSet_test[i], bag3_data[:], bag3_label, 7)
#         r4 = classify(dataSet_test[i], bag4_data[:], bag4_label, 7)
#         r5 = classify(dataSet_test[i], bag5_data[:], bag5_label, 7)
#         classCount[r1] = classCount.get(r1, 0) + 1 #default 0
#         classCount[r2] = classCount.get(r2, 0) + 1 #default 0
#         classCount[r3] = classCount.get(r3, 0) + 1 #default 0
#         classCount[r4] = classCount.get(r4, 0) + 1 #default 0
#         classCount[r5] = classCount.get(r5, 0) + 1 #default 0
#         sortedClassCount = sorted(classCount.items(), key=lambda d:d[1], reverse=True)
#         if(sortedClassCount[0][0] == labels_test[i]):
#             count += 1

#     print("ACC: ", count/(epoch)*100)

# def pure_method():
#     '''
#     没有加任何对不平衡数据处理的情况
#     epoch = 500, acc = 81.2
#     epoch = 2000, acc = 81.9
#     '''
#     dataSet, labels = TrainSet()
#     dataSet_test, labels_test = TestSet()
#     count = 0
#     epoch = 2000
#     for i in tqdm(range(epoch)):
#         r = classify(dataSet_test[i], dataSet[:], labels, 5)
#         if(r == labels_test[i]):
#             count += 1

#     print("ACC: ", count/(epoch)*100)


# def test():
#     Store_data = np.load('Store_data.npy')
#     Store_label = np.load('Store_label.npy')
#     dataSet_test, labels_test = TestSet()
#     n, _ = dataSet_test.shape
#     epoch = n   # 此时epoch表示测试集的数据条数
#     total1 = 0
#     TP1 = 0
#     FN1 = 0
#     total2 = 0
#     TP2 = 0
#     FN2 = 0
#     total3 = 0
#     TP3 = 0
#     FN3 = 0
#     for i in tqdm(range(epoch)):
#         if labels_test[i] == 1:
#             total1 += 1
#             r = classify(dataSet_test[i], Store_data[:], Store_label[:], 5)
#             if r == 1:
#                 TP1 += 1
#             elif r == 2:
#                 FN2 += 1 
#             elif r == 3:
#                 FN3 += 1
#         elif labels_test[i] == 2:
#             total2 += 1
#             r = classify(dataSet_test[i], Store_data[:], Store_label[:], 5)
#             if r == 2:
#                 TP2 += 1
#             elif r == 1:
#                 FN1 += 1
#             elif r == 3:
#                 FN3 += 1
#         elif labels_test[i] == 3:
#             total3 += 1
#             r = classify(dataSet_test[i], Store_data[:], Store_label[:], 5)
#             if r == 3:
#                 TP3 += 1
#             elif r == 1:
#                 FN1 += 1
#             elif r == 2:
#                 FN2 += 1
#     print("CLASS SMALL: ")
#     print("Precision = {:.2f}%".format(100*TP1/total1))
#     print("Recall = {:.2f}%".format(100*TP1/(TP1 + FN1)))
#     print("CLASS TRUE TO SIZE: ")
#     print("Precision = {:.2f}%".format(100*TP2/total2))
#     print("Recall = {:.2f}%".format(100*TP2/(TP2 + FN2)))
#     print("CLASS LARGE: ")
#     print("Precision = {:.2f}%".format(100*TP3/total1))
#     print("Recall = {:.2f}%".format(100*TP3/( TP3 + FN3)))
#     print("\nACC = {:.2f}%".format(100*(TP1+TP2+TP3)/(total1+total2+total3)))

