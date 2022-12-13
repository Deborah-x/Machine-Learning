import numpy as np
import pandas as pd
from tqdm import tqdm

def read_data():
    path = 'data_proc.txt'
    data = pd.read_csv(path)[['item_name','fit','height','weight']]
    # x1 = np.array(data['item_name'])
    # x2 = np.array(data['fit'])
    # x3 = np.array(data['height'])
    # x4 = np.array(data['weight'])
    # x5 = np.array(data['rating'])
    # print(np.array(data))
    # return x1, x2, x3, x4, x5, data
    return np.array(data)


def init():
    # x1, x2, x3, x4, data = read_data()
    data = read_data()
    data_s = data[2]    # 'fit' = 'Small'
    data_t = data[1]    # 'fit' = 'True to Size'
    data_l = data[3]    # 'fit' = 'Large'
    class_center = []
    class_center.append(data_s)
    class_center.append(data_t)
    class_center.append(data_l)
    return class_center

def distance(a, b):
    dis = np.sqrt(np.sum(np.square(a - b)))
    return dis

def dist_rank(center, data):
    # 得到与类中心最小距离的类别位置索引
    t = []
    for i in range(3):
        d = distance(data, center[i])
        t.append(d)
    loc = t.index(min(t))
    return loc

def means(arr):
    # 计算类的平均值当作类的新的中心
    sum = [0.0, 0.0, 0.0, 0.0, 0.0]
    for i in arr:
        sum += i
    mean = np.divide(sum, len(arr))
    return mean

def divide(center, data):
    # 将每个样本分到与之欧式距离最近的一个类里面
    cla_arr = [[], [], []]
    for i in range(len(data)):
        loc = dist_rank(center, data[i])
        cla_arr[loc].append(list(data[i]))
    return cla_arr

def new_center(cla):
    # 计算每个类平均值，并更新类中心
    new_cen = []
    for i in range(3):
    # print(np.mean(cla[0], axis=0))
    # print(np.mean(cla[1], axis=0))
    # print(np.mean(cla[2], axis=0))
        new = np.mean(cla[i], axis=0)
        new_cen.append(new)
    return new_cen

def Kmeans(n):
    # 获取n次更新之后的类别中心
    data = read_data()  # 读取数据
    center = init()     # 获取初始类别中心
    for _ in tqdm(range(n)):
        cla_arr = divide(center, data)  # 将数据分到选取的类中心里
        center = new_center(cla_arr)    # 更新类别中心
    return center    

def test(center):
    count = 0
    data_test = read_data()
    for i in tqdm(range(len(data_test))):
        if dist_rank(center, data_test[i]) == data_test[i][1]:
            count += 1
        # print(dist_rank(center, data_test[i]))

    return count / len(data_test)

# print(read_data()[:5])
# print(init())
# distance(init()[0], init()[1])
center = Kmeans(100)
print(test(center))

# data = read_data()
# center = init()
# cla_arr = divide(center, data)
# center = new_center(cla_arr)
# print(center)