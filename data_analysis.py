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

data_raw = load_dataset()
# with open('data_proc', 'w+') as f:
#     wf = csv.writer(f)
#     wf.writerows(data_raw)
# data_df = pd.DataFrame(data)
# data_df = pd.read_json("./train_data_all.json")
# data_fit = data_df.loc[data_df['fit']!='']
# print(data_fit.head())

# data_raw = np.array(data)
# np.save('data_raw.npy', data_raw)
# t = np.load('data_raw.npy', allow_pickle=True)
# t = t.tolist()
# print(t[1])
# np.savetxt('data_proc', data_raw, delimiter=',', newline='\n', fmt="%s", encoding='utf-8')
# data_proc = np.loadtxt('data_proc', dtype=list, unpack=True, encoding='utf-8')
# print(data_proc)
df = pd.DataFrame(data_raw)
df.to_csv('data_proc', index=False)
df_ = pd.read_csv('data_proc')
# print(df_.dropna())
df_ = df_.dropna()
print(df_)