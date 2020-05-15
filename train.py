from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from torch import from_numpy, tensor
import numpy as np
from network import Net
import os
'''
pytorch
opencv
'''

# Training settings
batch_size = 64

class DiabetesDataset(Dataset):

    def __init__(self):

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len

    def __load__(self, string, x, y):
        root = os.path.join('./dataset', string, string)
        self.x_data = from_numpy(np.load(os.path.join(root, x + '.npy')))
        self.y_data = from_numpy(np.load(os.path.join(root, y + '.npy')))
        self.len = self.x_data.shape[0]





model = Net()

optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)


def train(epoch):
    ########3-fold training set만
    for i in range(0, 17):
        sess_str = 's' + str(i).zfill(3) + '_glasses'
        dataset_left = DiabetesDataset()
        dataset_left.__load__(sess_str, 'left_x', 'y_val')
        dataset_right = DiabetesDataset()
        dataset_right.__load__(sess_str, 'right_x','y_val')
        dataset_head = DiabetesDataset()
        dataset_head.__load__(sess_str, 'headPose_x', 'y_val')
        #코어개수 확인 num_workers############### + batch size 결정
        train_loader_left = DataLoader(dataset=dataset_left,
                                  batch_size=32,
                                  shuffle=False,
                                  num_workers=2)
        train_loader_right = DataLoader(dataset=dataset_right,
                                  batch_size=32,
                                  shuffle=False,
                                  num_workers=2)
        train_loader_head = DataLoader(dataset=dataset_head,
                                  batch_size=32,
                                  shuffle=False,
                                  num_workers=2)
        model.train()
        for batch_idx, (data_h, target) in enumerate(train_loader_head):
            data_h, target = Variable(data_h), Variable(target)
            data_l = Variable(train_loader_left[batch_idx][0])
            data_r = Variable(train_loader_right[batch_idx][0])

            optimizer.zero_grad()
            output = model(data_h, data_l, data_r)
            loss = F.nll_loss(output, target)
            loss.backward()
            optimizer.step()
            if batch_idx % 10 == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data_r), len(train_loader_right.dataset),
                    100. * batch_idx / len(train_loader_right), loss.item()))


def test():
    model.eval()
    test_loss = 0
    correct = 0
    for data, target in test_loader:
        data, target = Variable(data, volatile=True), Variable(target)
        output = model(data)
        # sum up batch loss
        test_loss += F.nll_loss(output, target, size_average=False).data
        # get the index of the max log-probability
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


for epoch in range(1, 10):
    train(epoch)
    test()