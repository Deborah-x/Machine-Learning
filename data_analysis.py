import os
import csv
import json
import numpy as np
import pandas as pd

def load_dataset():
    path = "train_data_all.json"
    with open(path, 'r') as f:
        dataset = json.load(f)
        return dataset

def key_num(data, *args):
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

# print(key_num('fit', 'weight', 'height', 'size', 'usually_wear'))

def drop_nan():
    data_raw = load_dataset()
    pd.DataFrame(data_raw).to_csv('data_proc.txt', index=False)
    pd.read_csv('data_proc.txt').dropna().to_csv('data_proc.txt', index=False)
    # df = pd.read_csv('data_proc.txt')
    # return df

def convert_h():
    # 将身高从英尺转化为标准单位厘米, 保留两位小数
    # 1ft = 30.48cm
    # 1in =  2.54cm
    df = drop_nan()
    df = pd.read_csv('data_proc.txt')
    df['height'] = df['height'].map(lambda x: round(int(x[0]) * 30.48 + int(x[3]) * 2.54, 2))
    df.to_csv('data_proc.txt', index=False)
    print(df['height'])

convert_h()