import os
import re
import csv
import json
import numpy as np
import pandas as pd


def load_dataset():
    # 加载训练数据集train_data_all.json，返回值为list类型
    path = "train_data_all.json"
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

def drop_nan(spath, dpath):
    # 将训练数据集中有空信息的数据扔掉，剩余数据放入一个新文件
    data = pd.read_csv(spath)
    # values = {'height': "NAN", 'weight': "140LBS"}
    # data.fillna(value=values, inplace=True)
    data.dropna().to_csv(dpath, index=False)

def convert_h(path):
    # 将身高从英尺转化为标准单位厘米, 保留两位小数
    # 1ft = 30.48cm
    # 1in =  2.54cm
    df = pd.read_csv(path)
    column = df['height']
    labels = df['fit']
    column = column.map(lambda x: round(int(x[0]) * 30.48 + int(x[3]) * 2.54, 2) if re.match(r'\d\' \d\"', str(x)) else 164) # 162大概是身高的平均水平
    # av_label = {}
    # av_label[0] = column[labels == 0].mean(skipna=True)
    # av_label[1] = column[labels == 1].mean(skipna=True)
    # av_label[2] = column[labels == 2].mean(skipna=True)
    # for i in range(len(column)):
    #     if column[i] == 0:
    #         column[i] = av_label[labels[i]]
    df['height'] = column
    df.to_csv(path, index=False)
    # print(df['height'])

def convert_w(path):
    df = pd.read_csv(path)
    column = df['weight']
    labels = df['fit']
    column = column.map(lambda x: int(x.strip('LBS')) if re.match(r'\d{2,3}LBS', x) else 140) # 140是体重的平均水平
    # for i in range(len(df['weight'])):
    #     if re.match(r'\d{2,3}LBS', df['weight'][i]):
    #         df['weight'][i] = int(df['weight'][i].strip('LBS'))
    #     else:
    #         df['weight'][i] = 140
    # av_label = {}
    # av_label[0] = column[labels == 0].mean(skipna=True)
    # av_label[1] = column[labels == 1].mean(skipna=True)
    # av_label[2] = column[labels == 2].mean(skipna=True)
    # for i in range(len(column)):
    #     if column[i] == 0:
    #         column[i] = av_label[labels[i]]
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

def val_range(path, key):
    # 统计数据集中关键词的范围
    df = pd.read_csv(path)
    return set(df[key])

def fill(path):
    df = pd.read_csv(path)
    values = {'height': "NAN", 'weight': "140LBS"}
    df.fillna(value=values, inplace=True)
    df.to_csv(path, index=False)

def proc():
    spath = 'data_raw.txt'
    dpath = 'data_proc.txt'
    drop_nan(spath, dpath)
    convert_h(dpath)
    convert_w(dpath)
    convert_f(dpath)
    convert_n(dpath)
    # convert_r(dpath)
    # convert_h(dpath)
    # convert_w(dpath)


if __name__ == '__main__':
    # load_dataset()
    # build_data_raw()
    spath = 'data_raw.txt'
    dpath = 'data_proc.txt'
    print(val_range(spath, 'body_type'))
    # proc()
    # print(val_range(spath, 'weight'))
    # data = pd.read_csv(dpath)[['item_name','height','weight','rating','fit']]
    # print(data)
    # print(data[(data['fit'] == 0)].size)
    # print(data[(data['fit'] == 1)].size)
    # print(data[(data['fit'] == 2)].size)
    # data_t = (data - data.mean())/data.std()
    # data_t['fit'] = data['fit']

    # data_t.to_csv(dpath, index=False)

    # path0 = 'data_fit_0.txt'
    # path1 = 'data_fit_1.txt'
    # path2 = 'data_fit_2.txt'
    # data_t[data_t['fit']==0].to_csv(path0)
    # data_t[data_t['fit']==1].to_csv(path1)
    # data_t[data_t['fit']==2].to_csv(path2)