import re
import json
import numpy as np
import pandas as pd
from tqdm import tqdm


def load_dataset():
    '''
    加载训练数据集train_data_all.json，返回值为list类型
    '''
    path = "train_data_all.json"
    with open(path, 'r') as f:
        dataset = json.load(f)
        return dataset

def build_data_raw():
    '''
    将数据集转化为dataframe类型并放入data_raw.txt
    '''
    pd.DataFrame(load_dataset()).to_csv('data_raw.txt', index=False)

def drop_nan(spath, dpath):
    '''
    将训练数据集中有空信息的数据扔掉，剩余数据放入一个新文件
    '''
    data = pd.read_csv(spath)
    data.dropna().to_csv(dpath, index=False)

def val_range(path, key):
    '''
    统计数据集中关键词的范围，返回值为set类型
    '''
    df = pd.read_csv(path)
    # print(df.shape)
    return set(df[key])

def proc():
    build_data_raw()
    drop_nan('data_raw.txt', 'data_proc.txt')

def build_list():
    dpath = "data_proc.txt"
    item_name_list = list(val_range(dpath, 'item_name'))
    user_name_list = list(val_range(dpath, 'user_name'))
    rented_for_list = list(val_range(dpath, 'rented_for'))
    usually_wear_list = list(val_range(dpath, 'usually_wear'))
    size_list = list(val_range(dpath, 'size'))
    age_list = list(val_range(dpath, 'age'))
    height_list = list(val_range(dpath, 'height'))
    bust_size_list = list(val_range(dpath, 'bust_size'))
    weight_list = list(val_range(dpath, 'weight'))
    body_type_list = list(val_range(dpath, 'body_type'))
    rating_list = list(val_range(dpath, 'rating'))
    price_list = list(val_range(dpath, 'price'))
    # print(len(item_name_list)+len(user_name_list)+len(rented_for_list)+len(usually_wear_list)+len(size_list)+len(age_list)+len(height_list)+len(bust_size_list)+len(weight_list)+len(body_type_list)+len(rating_list)+len(price_list))
    voca_list = item_name_list+user_name_list+rented_for_list+usually_wear_list+size_list+age_list+height_list+bust_size_list+weight_list+body_type_list+rating_list+price_list 
    return voca_list

def data2vector():
    voca_list = build_list()
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


if __name__ == "__main__":
    # proc()
    # data2vector()
    path ='data_proc.txt'
    data = pd.read_csv(path)
    label = data['fit']
    label = label.replace(['Small', 'True to Size', 'Large'], [0, 1, 2])
    np.save('label.npy', label.to_numpy())
    y = np.load('label.npy')
    print(y)