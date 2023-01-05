import torch
import numpy as np
from tqdm import tqdm
from torchvision import transforms
from torchvision import datasets
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.optim as optim

labels = np.load('label_mix.npy')
dataSet = np.load('data_mix.npy')
m, n = dataSet.shape

# print(dataSet.shape)  # (22071,4981),数据均衡后(45000,4981)
pad = np.zeros((m, 1419))
dataSet = np.concatenate((dataSet, pad), axis=1).reshape((m,80,80))
# print(dataSet.shape)  # (22071,6400),数据均衡后(45000,6400)
# print(dataSet.shape)  # (22071,80,80),数据均衡后(45000,80,80)
labels = torch.tensor(labels, dtype=torch.long)
dataSet = torch.tensor(dataSet, dtype=torch.float)
labels = labels-1
# design model using class


class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = torch.nn.Conv2d(1,5,kernel_size=5, padding=2)
        self.conv2 = torch.nn.Conv2d(5,10,kernel_size=5, padding=2)
        self.conv3 = torch.nn.Conv2d(10,5,kernel_size=5, padding=2)
        self.conv4 = torch.nn.Conv2d(5,5,kernel_size=5, padding=2)
        self.pooling = torch.nn.MaxPool2d(2)
        self.fc = torch.nn.Linear(125,3)   # fc:full connected layer 全连接层

    def forward(self, x):
        # Flatten data from (n,1,28,28) to (n,784)
        batch_size = x.size(0)
        x = F.relu(self.pooling(self.conv1(x)))
        x = F.relu(self.pooling(self.conv2(x)))
        x = F.relu(self.pooling(self.conv3(x)))
        x = F.relu(self.pooling(self.conv4(x)))
        x = x.view(batch_size,-1)   # flatten
        x = self.fc(x)
        return x
   
model = Net()

# construct loss and optimizer
criterion = torch.nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.5)

# training cycle forward, backward, update
def train(epoch):
    for i in range(epoch):
        for j in tqdm(range(m)):
            # 获得一个批次的数据和标签
            inputs = dataSet[j:j+1]
            target = labels[j:j+1]
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, target)
            loss.backward()
            optimizer.step()
    


def test(epoch):
    correct = 0
    total = 0
    with torch.no_grad():
        for u in tqdm(range(epoch)):
            images = dataSet[u:u+1]
            label = labels[u:u+1]
            outputs = model(images)
            _, predicted = torch.max(outputs.data, dim=1) # dim = 1 列是第0个维度，行是第1个维度
            total += label.size(0)
            correct += (predicted == label).sum().item() # 张量之间的比较运算
    print('accuracy on test set: %d %% ' % (100*correct/total))
 
 
if __name__ == '__main__':
    train(2)
    test(m)
    # print(model)
    pass