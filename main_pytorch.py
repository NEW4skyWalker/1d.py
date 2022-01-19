# -*- coding: utf-8 -*-
"""
Created on Mon Dec 20 19:06:32 2021

"""

import numpy as np
import torch
from torch.utils.data import DataLoader,Dataset
from torchvision import transforms
import gzip
import os
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

###参数
n_epochs = 3
batch_size_train = 64
batch_size_test = 1000
learning_rate = 0.01
momentum = 0.5
log_interval = 10
random_seed = 1
torch.manual_seed(random_seed)


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class DealDataset(Dataset):
    """
        读取数据、初始化数据
    """
    def __init__(self, folder, data_name, label_name,transform=None):
        (train_set, train_labels) = load_data(folder, data_name, label_name) # 其实也可以直接使用torch.load(),读取之后的结果为torch.Tensor形式
        self.train_set = train_set
        self.train_labels = train_labels
        self.transform = transform

    def __getitem__(self, index):

        img, target = self.train_set[index], int(self.train_labels[index])
        if self.transform is not None:
            img = self.transform(img)
        return img, target

    def __len__(self):
        return len(self.train_set)
  
def load_data(data_folder, data_name, label_name):
    with gzip.open(os.path.join(data_folder,label_name), 'rb') as lbpath: # rb表示的是读取二进制数据
        y_train = np.frombuffer(lbpath.read(), np.uint8, offset=8)

    with gzip.open(os.path.join(data_folder,data_name), 'rb') as imgpath:
        x_train = np.frombuffer(
            imgpath.read(), np.uint8, offset=16).reshape(len(y_train), 28, 28)
    return (x_train, y_train)

trainDataset = DealDataset('E:/研究生/研究生课程资料/网络测量/udp', "train-images-idx3-ubyte.gz","train-labels-idx1-ubyte.gz",transform=transforms.ToTensor())
testDataset = DealDataset('E:/研究生/研究生课程资料/网络测量/udp', "t10k-images-idx3-ubyte.gz","t10k-labels-idx1-ubyte.gz",transform=transforms.ToTensor())
# 训练数据和测试数据的装载
train_loader = torch.utils.data.DataLoader(
    dataset=trainDataset,
    batch_size=64, 
    shuffle=False,
)

###测试数据
test_loader = torch.utils.data.DataLoader(
    dataset=testDataset,
    batch_size=64, 
    shuffle=False,
)

# examples = enumerate(train_loader)
# batch_idx, (example_data, example_targets) = next(examples)
# print(example_targets) ###随机输出训练样本标签
# print(example_data.shape)
#
#
# examples = enumerate(test_loader)
# batch_idx, (example_data, example_targets) = next(examples)
# print(example_targets) ###随机输出训练样本标签
# print(example_data.shape)


# import matplotlib.pyplot as plt
# fig = plt.figure()
# for i in range(6):
#   plt.subplot(2,3,i+1)
#   plt.tight_layout()
#   plt.imshow(example_data[i][0], cmap='gray', interpolation='none')
#   plt.title("Ground Truth: {}".format(example_targets[i]))
#   plt.xticks([])
#   plt.yticks([])
# plt.show()

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim



class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=5, stride=1, padding=2)
        self.conv2 = nn.Conv2d(64, 32, kernel_size=5, stride=1, padding=2)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(7*7*32, 50)
        self.fc2 = nn.Linear(50, 2)
    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 7*7*32)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x)


network = Net().to(device)
optimizer = optim.SGD(network.parameters(), lr=learning_rate,
                      momentum=momentum)


train_losses = []
train_counter = []
test_losses = []
test_counter = [i*len(train_loader.dataset) for i in range(n_epochs + 1)]

def train(epoch):
  network.train()
  for epoch in range(epoch):
      for batch_idx, (data, target) in enumerate(train_loader):
        data=data.to(device)
        target=target.to(device)
        optimizer.zero_grad()
        output = network(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % log_interval == 0:
          print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
            epoch, batch_idx * len(data), len(train_loader.dataset),
            100. * batch_idx / len(train_loader), loss.item()))
          train_losses.append(loss.item())
          train_counter.append(
            (batch_idx*64) + ((epoch-1)*len(train_loader.dataset)))
          torch.save(network.state_dict(), './model.pth')

          
train(100)


def test():
  network.load_state_dict(torch.load('./model.pth'))
  network.eval()
  network.to(device)
  test_loss = 0
  correct = 0
  list1=[]
  list2=[]

  with torch.no_grad():
    for data, target in test_loader:
        data=data.to(device)
        target=target.to(device)
        target1=target
        list1.extend(target1.cpu().numpy())

        output = network(data)
        test_loss += F.nll_loss(output, target, size_average=False).item()
        pred = output.data.max(1, keepdim=True)[1]
        pred1=pred
        list2.extend(pred1.cpu().numpy())
        correct += pred.eq(target.data.view_as(pred)).sum()
    test_loss /= len(test_loader.dataset)
    test_losses.append(test_loss)
    print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.4f}%)\n'.format(
    test_loss, correct, len(test_loader.dataset),
    100. * correct / len(test_loader.dataset)))
    print(' report is ',classification_report(list1, list2,digits=8))

test()
