import os
import csv
import json
import numpy as np
import pandas as pd


def load_dataset():
    # 加载训练数据集
    path = "train_data_all.json"
    with open(path, 'r') as f:
        dataset = json.load(f)
        return dataset


def key_num(data, *args):
    # 统计所给关键词都不为空的训练数据数量
    print(args)
    count = 0
    for index in data:
        flag = 1
        for key in args:
            if index[key] == '':
                flag = 0
                break
        count += flag
    return count


def drop_nan():
    # 将训练数据集中有空信息的数据扔掉，剩余数据放入一个新文件
    data_raw = load_dataset()
    pd.DataFrame(data_raw).to_csv('data_proc.txt', index=False)
    pd.read_csv('data_proc.txt').dropna().to_csv('data_proc.txt', index=False)
    df = pd.read_csv('data_proc.txt')
    return df


def convert_h():
    # 将身高从英尺转化为标准单位厘米, 保留两位小数
    # 1ft = 30.48cm
    # 1in =  2.54cm
    df = drop_nan()
    df = pd.read_csv('data_proc.txt')
    df['height'] = df['height'].map(lambda x: round(int(x[0]) * 30.48 + int(x[3]) * 2.54, 2))
    df.to_csv('data_proc.txt', index=False)
    print(df['height'])


def val_range(key):
    # 统计关键词的范围
    df = pd.DataFrame(load_dataset())
    return set(df[key])

# convert_h()
# df = pd.read_csv('data_proc.txt')
print(val_range('rating'))