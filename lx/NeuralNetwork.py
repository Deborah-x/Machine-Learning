import torch
import numpy as np
import pandas as pd

path = "data_proc.txt"
data = pd.read_csv(path)

data = torch.from_numpy(data.values)   # 输入大小(22071, 5) 前四项是数据，最后一项是标签
data = data.float()
# print(data)

class LinearModel(torch.nn.Module):
    def __init__(self):#构造函数
        super(LinearModel,self).__init__()
        self.linear = torch.nn.Linear(4,1)#构造对象，并说明输入输出的维数，第三个参数默认为true，表示用到b
    def forward(self, x):
        y_pred = self.linear(x)#可调用对象，计算y=wx+b
        return  y_pred

model = LinearModel()#实例化模型
criterion = torch.nn.MSELoss(reduction='sum')
#model.parameters()会扫描module中的所有成员，如果成员中有相应权重，那么都会将结果加到要训练的参数集合上
optimizer = torch.optim.ASGD(model.parameters(),lr=0.01)#lr为学习率
# 将torch.optim.SGD换成Adagrad Adam adamax ASGD RMSprop Rprop运行比较

def train(epoch):
    # running_loss = 0.0
    for idx, val in enumerate(data, 0):
        x_data = val[0:4]
        y_data = val[4]
        y_pred = model(x_data)
        optimizer.zero_grad()
        loss = criterion(torch.squeeze(y_pred),y_data)
        loss.backward()
        optimizer.step()
        # running_loss += loss.item()
        # if idx % 100 == 99:
        #     print('[%d, %d] loss: %.3f' % (epoch+1, idx+1, running_loss/100))
        #     running_loss = 0.0

def pred(data):
    if data < 0.5: 
        return 0
    if data >= 0.5 and data < 1.5:
        return 1
    if data >= 1.5:
        return 2

def test():
    correct = 0
    for i in range(1000):
        test = data[i]
        x_test = test[0:4]
        y_test = test[4]
        y_pred = pred(model(x_test).data)
        if y_pred == y_test:
            correct += 1
    print("Accuracy: ", correct / 1000 * 100, "%")


if __name__ == '__main__':
    train(1)
    test()
    print('w=',model.linear.weight)
    print('b=',model.linear.bias)

    
