import os
import re
import csv
import json
import numpy as np
import pandas as pd
import tqdm


def load_dataset():
    # 加载训练数据集train_data_all.json，返回值为list类型
    path = "C:/Users/86189/PycharmProjects/MLproject/Machine-Learning/train_data_all.json"
    with open(path, 'r') as f:
        dataset = json.load(f)
        return dataset

def build_data_raw():
    # 将数据集转化为dataframe类型并放入data_raw.txt
    pd.DataFrame(load_dataset()).to_csv('data_raw.txt', index=False)

# def key_num(data, *args):
#     # 统计所给关键词都不为空的训练数据数量
#     print(args)
#     count = 0
#     for index in data:
#         flag = 1
#         for key in args:
#             if index[key] == '':
#                 flag = 0
#                 break
#         count += flag
#     return count



def get_review(spath, dpath):
    data = pd.read_csv(spath)[['review','fit']]
    data.to_csv('data_review.txt', index=False)

def convert_h(path):
    # 将身高从英尺转化为标准单位厘米, 保留两位小数
    # 1ft = 30.48cm
    # 1in =  2.54cm
    df = pd.read_csv(path)
    column = df['height']
    labels = df['fit']
    column = column.map(lambda x: round(int(x[0]) * 30.48 + int(x[3]) * 2.54, 2) if re.match(r'\d\' \d\"', str(x)) else 164) # 162大概是身高的平均水平
    av_label = {}
    av_label[0] = column[labels == 0].mean(skipna=True)
    av_label[1] = column[labels == 1].mean(skipna=True)
    av_label[2] = column[labels == 2].mean(skipna=True)
    for i in range(len(column)):
        if column[i] == 0:
            column[i] = av_label[labels[i]]
    df['height'] = column
    df.to_csv(path, index=False)
    # print(df['height'])

def convert_w(path):
    df = pd.read_csv(path)
    column = df['weight']
    labels = df['fit']
    column = column.map(lambda x: int(x.strip('LBS')) if re.match(r'\d{2,3}LBS', x) else 140) # 140是体重的平均水平
    av_label = {}
    av_label[0] = column[labels == 0].mean(skipna=True)
    av_label[1] = column[labels == 1].mean(skipna=True)
    av_label[2] = column[labels == 2].mean(skipna=True)
    for i in range(len(column)):
        if column[i] == 0:
            column[i] = av_label[labels[i]]
    df['weight'] = column
    df.to_csv(path, index=False)
    # print(df['weight'])

def convert_f(path):
    data = pd.read_csv(path)
    data['fit'] = data['fit'].replace(['Small', 'True to Size', 'Large'], [0, 1, 2])
    data.to_csv(path, index=False)

def convert_n(path):
    data = pd.read_csv(path)
    uni = data['item_name'].unique()
    # new_list = [10*x for x in list(range(len(uni)))]
    # map = dict(zip(uni, new_list))
    map = dict(zip(uni, list(range(len(uni)))))
    # print(map)
    data['item_name'] = data['item_name'].replace(map)
    data.to_csv(path, index=False)

def convert_r(path):
    data = pd.read_csv(path)
    # data['rating'] = data['rating'].replace([0, 1, 2, 3, 4, 5], [0, 10, 20, 30, 40, 50])
    data.to_csv(path, index=False)

def proc():
    spath = 'data_raw.txt'
    dpath = 'data_proc.txt'
    #data = pd.read_csv(spath)
    drop_nan(spath, dpath)
    #get_review(spath,dpath)
    # convert_h(dpath)
    # convert_w(dpath)
    # convert_f(dpath)
    # convert_n(dpath)
    # convert_r(dpath)
    # convert_h(dpath)
    # convert_w(dpath)

def val_range(path, key):
    '''
    统计数据集中关键词的范围，返回值为set类型
    '''
    df = pd.read_csv(path)
    # print(df.shape)
    return set(df[key])

def delete_len1(list):
   for i in range(len(list) - 1, -1, -1):
      if len(list[i]) == 1 and not list[i].isdigit():
         del list[i]

def split2word(words):
   word = []
   for i in range(len(words)):
      word += words[i].split()
   delete_len1(word)
   return word

def build_list():
    f = open(r"C:/Users/86189/PycharmProjects/MLproject/Machine-Learning/cyb/data_proc.txt", 'r', encoding="utf-8")
    set = f.readlines()
    voca_list = []
    for i in range(len(set)):
        words = split2word(set[i])
        voca_list += words
    return list(set(voca_list))

def data2vector():
    voca_list = build_list()
    print("向量长度为{}".format(len(voca_list)))
    dataset = pd.read_csv('data_proc.txt')
    m, n = dataset.shape
    # X = [0]*len(voca_list)
    feature = ['item_name', 'user_name', 'rented_for', 'usually_wear', 'size', 'age', 'height', 'bust_size', 'weight', 'body_type', 'rating', 'price']
    X = np.zeros((m, len(voca_list)))
    for i in tqdm(range(m)):
        x = dataset.iloc[i]
        for j in feature:
            X[i, voca_list.index(x[j])] = 1

    np.save('data2vector.npy', X)

def getnum_eachkind():
    data = pd.read_csv('data_proc.txt')
    num_t2s = 0
    num_s = 0
    num_l = 0
    n, m = data.shape
    for i in range(n):
        if data.iat[i,2] == "True to Size":
            num_t2s += 1
        if data.iat[i,2] == "Small":
            num_s += 1
        if data.iat[i,2] == "Large":
            num_l += 1
    print("len of t2s is {}".format(num_t2s))
    print("len of s is {}".format(num_s))
    print("len of l is {}".format(num_l))

def drop_nan(spath, dpath):
    # 将训练数据集中有空信息的数据扔掉，剩余数据放入一个新文件
    data = pd.read_csv(spath, usecols = ['item_name','price','fit','rented_for','usually_wear','size','age','height','bust_size','weight','body_type'])
    #data = pd.read_csv(spath)
    df = data.dropna()
    m, n = df.shape
    for i in range(m):
        for j in range(n):
            x = df.iat[i,j]
            df.iat[i,j] = str(x).replace('\n',' ')
    df.to_csv(dpath, index=False)




spath = 'data_raw.txt'
dpath = 'data_proc.txt'
proc()
data = pd.read_csv(dpath)
#print(len(data))
getnum_eachkind()
#全数据下
# len of t2s is 16126
# len of s is 2594
# len of l is 3351
#test数据下
# len of t2s is 16303
# len of s is 2660
# len of l is 3413
# print(val_range(dpath, 'fit'))
# data = pd.read_csv(dpath)[['item_name','height','weight','rating','fit']]
# print(data.size)
# print(data[(data['fit'] == 0)].size)
# print(data[(data['fit'] == 1)].size)
# print(data[(data['fit'] == 2)].size)
# data_t = (data - data.mean())/data.std()
# data_t['fit'] = data['fit']
# data_t.to_csv(dpath, index=False)