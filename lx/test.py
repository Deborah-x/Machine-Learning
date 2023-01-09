import torch
from torch import nn
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from data_proc import *
from sklearn.metrics import f1_score,precision_score,recall_score

spath = 'train_data_all.json'
with open(spath, 'r', encoding="utf-8") as f:
    dataset = json.load(f)  # type 'list'

# Convert the training set to type 'dataframe' and put them into a file named dpath
dpath = 'train_proc.txt'
pd.DataFrame(dataset).to_csv(dpath, index=False)
# data = pd.read_json(spath).to_csv(dpath, index=False)
# Discard the data with empty 'fit' index in the training set and put the remaining data into the old file
data = pd.read_csv(dpath)
data = data.dropna(subset=['fit'])
data['fit'].replace(['Small', 'True to Size', 'Large'], [1, 2, 3], inplace=True)
print(data)
with open('test.json', 'w', encoding="utf-8") as f:
    f.write(data.to_json(orient='records'))
# with open('test.json', 'r', encoding="utf-8") as f:
    # dataset = json.load(f)  # type 'list'
# print(dataset)
# #三分类问题
# trueY=np.matrix([[1,2,3,2,1,3,1,3,1,1,3,2,3,2]]).T
# testY=np.matrix([[1,2,3,2,2,3,1,3,1,1,3,2,3,2]]).T

# oriF1=f1_score(trueY,testY,average="macro")
# print("sklearn-f1:",oriF1)

# f1_1=f1_score(y_true = (trueY==1), y_pred = (testY==1))#针对分类1的f1
# f1_2=f1_score(y_true = (trueY==2), y_pred = (testY==2))#针对分类2的f1
# f1_3=f1_score(y_true = (trueY==3), y_pred = (testY==3))#针对分类3的f1
# f1_123=np.mean([f1_1,f1_2,f1_3])#计算均值

# print("ave-f1:",f1_123)

# group0 = pd.read_csv("data_fit_0.txt")
# group1 = pd.read_csv("data_fit_1.txt")
# group2 = pd.read_csv("data_fit_2.txt")
# dataSet0 = np.array(group0[['item_name','height','weight','rating']])
# dataSet1 = np.array(group1[['item_name','height','weight','rating']])
# dataSet2 = np.array(group2[['item_name','height','weight','rating']])
# labels0 = np.array(group0['fit'])
# labels1 = np.array(group1['fit'])
# labels2 = np.array(group2['fit'])

# print(np.concatenate((labels0[:10], labels1[:10], labels2[:10]), axis=0))
# x = np.zeros((3,4))
# x[1,2] = 1
# np.save('test.npy', x)
# y = np.load('test.npy')
# # print(x)
# # print(y)
# X = np.random.randint(0, 5, size=(2,3))
# print(X)
# X[1,1] = 10
# print(X)

# print(np.square(X))
# print(np.sum(np.square(X)))

# x = np.arange(3,10)
# y = np.arange(3,10)
# acc = np.load('acc.npy')
# acc = acc.reshape(7,7)
# print(acc)
# plt.figure()
# for i in range(7):
#     # plt.figure()
#     plt.plot(x,acc[i])
#     # plt.show()
# plt.show()

# x = np.array([[1,2,3],[4,5,6]])
# y = np.zeros((2,3))
# x = np.concatenate((x,y), axis=1)
# print(x)

# labels = np.load('label2vector_wide.npy')
# dataSet = np.load('data2vector_wide.npy')
# # print(dataSet.shape)  # (22071,4981)
# pad = np.zeros((22071, 1419))
# dataSet = np.concatenate((dataSet, pad), axis=1).reshape((22071,80,80))
# # print(dataSet.shape)  # (22071,6400)
# # print(dataSet.shape)  # (22071,80,80)
# labels = torch.tensor(labels)
# dataSet = torch.tensor(dataSet)
# batch_size = 21
# i = 0
# inputs = dataSet[i*batch_size:(i+1)*batch_size]
# target = labels[i*batch_size:(i+1)*batch_size]

# print(inputs.shape)
# print(target.shape)

# labels = np.load('label2vector_wide.npy')
# dataSet = np.load('data2vector_wide.npy')
# m, n = dataSet.shape
# total1 = 0
# total2 = 0
# total3 = 0
# index1 = []
# index2 = []
# index3 = []
# for i in range(m):
#     if labels[i] == 1:
#         index1.append(i)
#     elif labels[i] == 2:
#         index2.append(i)
#     elif labels[i] == 3:
#         index3.append(i)
# data1 = dataSet[index1]
# label1 = labels[index1]
# data2 = dataSet[index2]
# label2 = labels[index2]
# data3 = dataSet[index3]
# label3 = labels[index3]
# np.save('data1.npy', data1)
# np.save('data2.npy', data2)
# np.save('data3.npy', data3)
# np.save('label1.npy', label1)
# np.save('label2.npy', label2)
# np.save('label3.npy', label3)
# index1 = index1 * 6
# index3 = index3 * 5
# print(len(index1))
# print(len(index2))
# print(len(index3))
# index_mix = []
# for i in tqdm(range(15000)):
#     index_mix.append(index1[i])
#     index_mix.append(index2[i])
#     index_mix.append(index3[i])
# print(len(index_mix))
# data_mix = dataSet[index_mix]
# label_mix = labels[index_mix]
# np.save('data_mix.npy', data_mix)
# np.save('label_mix.npy', label_mix)

# embedding = nn.Embedding(5000, 1000) # 4096 = 64*64
# data = np.load('data2embedding.npy')
# m, n = data.shape
# print(data.shape)
# # for i in tqdm(range(m)):
# dataset = embedding(torch.LongTensor(data))
#     # if i == 0:
#     #     dataset = d
#     # else:
#     #     dataset = torch.cat((dataset, d), dim=0)
    

# print(embedding)
# print(dataset.shape)
