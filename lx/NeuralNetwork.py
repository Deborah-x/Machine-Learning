import torch
import numpy as np
import pandas as pd
from tqdm import tqdm

path = "data_proc.txt"
label = np.load('label2vector.npy')
# label = torch.tensor(label, dtype=torch.float32)
data = np.load('data2vector.npy')
# data = torch.tensor(data, dtype=torch.float32)
m, n = data.shape

w = np.random.rand(n,1)
# b = np.random.rand(1,1)
# b = 0
print("w0 = ", w)
# print("b0 = ", b)

def forward(x):
    # return np.dot(x, w) + b
    return np.dot(x, w)

def cost(X, Y):
    Y_pred = np.zeros(m,1)
    for i in range(m):
        Y_pred[i] = forward(X[i])
    loss = np.sum(np.square(Y_pred - Y))
    return loss / m

def gradient(x, y):
    delta_w = 2 * np.transpose(x) * (np.dot(x, w) - y)
    # delta_b = 2 * (np.dot(x, w) - y)
    # print(delta_w.shape)
    # print(delta_b.shape)
    return delta_w

def closer(pos):
    if pos <= -0.5:
        return -1
    elif pos >= 0.5:
        return 1
    else:
        return 0


if __name__ == '__main__':
    '''
    换成12维的数据集之后需要归一化，不然会超范围
    '''
    # lr = 0.0001
    # for epoch in tqdm(range(5000)):
    #     for i in range(m):
    #         delta_w = gradient(data[i].reshape(1,n), label[i].reshape(1,1))
    #         w -= lr * delta_w
    #         # b -= lr * delta_b
    # print("w = ", w)
    # # print("b = ", b)
    # np.save('w.npy', w)

    w = np.load('w.npy')
    num = 10000
    count = 0
    for i in tqdm(range(num)):
        if closer(forward(data[i].reshape(1,n))) == label[i]:
            count += 1
    print("Acc = ", count*100/num, "%")