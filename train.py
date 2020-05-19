from __future__ import print_function

import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch import from_numpy
from torch.autograd import Variable
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.data.distributed import DistributedSampler


# 3-fold
training_set = [[1, 2, 8, 10], [3, 4, 7, 9]]
test_set = [5, 6, 11, 12, 13]
validation_set = [14, 15, 16]

norm_mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
norm_std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

def train(epoch, model, batch_size, num_workers, gpu):
    train_loss = nn.MSELoss().cuda(gpu)
    validation_loss = nn.MSELoss(reduction='sum').cuda(gpu)
    optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.95))
    for s in training_set:
        for i in s:
            sess_str = 's' + str(i).zfill(3) + '_glasses'

            root = os.path.join('./dataset', sess_str, sess_str)
            dataset = TensorDataset(from_numpy(np.load(os.path.join(root, 'left_x.npy'))),
                                    from_numpy(np.load(os.path.join(root, 'right_x.npy'))),
                                    from_numpy(np.load(os.path.join(root, 'headPose_x.npy'))),
                                    from_numpy(np.load(os.path.join(root, 'y_val.npy'))))
            sampler = DistributedSampler(dataset)
            # Cores checking : num_workers, batch size
            train_loader = DataLoader(dataset=dataset,
                                      batch_size=batch_size,
                                      shuffle=False,
                                      num_workers=num_workers,
                                      pin_memory=True,
                                      sampler=sampler)

            model.train()
            for batch_idx, (data_l, data_r, data_h, target) in enumerate(train_loader):
                data_l = (data_l - norm_mean) / norm_std
                data_r = (data_r - norm_mean) / norm_std
                data_l, data_r, data_h, target = Variable(data_l.cuda(gpu)), Variable(data_r.cuda(gpu)), Variable(data_h.cuda(gpu)), Variable(target.cuda(gpu))
                optimizer.zero_grad()
                output = model(data_l.float(), data_r.float(), data_h.float())
                loss = train_loss(output, target.float())
                loss.backward()
                optimizer.step()
                '''
                if batch_idx % 10 == 0:
                    print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        epoch, batch_idx * len(data_r), len(train_loader.dataset),
                        100. * batch_idx / len(train_loader), loss.item()))
                '''
            #print('\n')

        # Validation
        final_loss = 0
        length = 0
        for i in validation_set:
            sess_str = 's' + str(i).zfill(3) + '_glasses'

            root = os.path.join('./dataset', sess_str, sess_str)
            dataset = TensorDataset(from_numpy(np.load(os.path.join(root, 'left_x.npy'))),
                                    from_numpy(np.load(os.path.join(root, 'right_x.npy'))),
                                    from_numpy(np.load(os.path.join(root, 'headPose_x.npy'))),
                                    from_numpy(np.load(os.path.join(root, 'y_val.npy'))))
            sampler = DistributedSampler(dataset)
            # Cores checking : num_workers, batch size
            validation_loader = DataLoader(dataset=dataset,
                                           batch_size=batch_size,
                                           shuffle=False,
                                           num_workers=num_workers,
                                           pin_memory=True,
                                           sampler=sampler)

            model.eval()
            loss = 0
            with torch.no_grad():
                for batch_idx, (data_l, data_r, data_h, target) in enumerate(validation_loader):
                    data_l, data_r, data_h, target = Variable(data_l.cuda(gpu)), Variable(data_r.cuda(gpu)), Variable(data_h.cuda(gpu)), Variable(target.cuda(gpu))
                    output = model(data_l.float(), data_r.float(), data_h.float())
                    loss += validation_loss(output, target.float()).data
                final_loss += loss
                length += len(validation_loader.dataset)
    print('\nValidation set: Epochs : {}\t Average loss: {:.4f}\n'.format(epoch, final_loss/length))
    return model, final_loss/length


def test(model, batch_size, num_workers, gpu):
    test_loss = nn.MSELoss(reduction='sum').cuda(gpu)
    # Test
    final_loss = 0
    length = 0
    for i in test_set:
        sess_str = 's' + str(i).zfill(3) + '_glasses'

        root = os.path.join('./dataset', sess_str, sess_str)
        dataset = TensorDataset(from_numpy(np.load(os.path.join(root, 'left_x.npy'))),
                                from_numpy(np.load(os.path.join(root, 'right_x.npy'))),
                                from_numpy(np.load(os.path.join(root, 'headPose_x.npy'))),
                                from_numpy(np.load(os.path.join(root, 'y_val.npy'))))
        sampler = DistributedSampler(dataset)
        # Cores checking : num_workers, batch size
        test_loader = DataLoader(dataset=dataset,
                                 batch_size=batch_size,
                                 shuffle=False,
                                 num_workers=num_workers,
                                 pin_memory=True,
                                 sampler=sampler)

        model.eval()
        loss = 0
        with torch.no_grad():
            for batch_idx, (data_l, data_r, data_h, target) in enumerate(test_loader):
                data_l, data_r, data_h, target = Variable(data_l.cuda(gpu)), Variable(data_r.cuda(gpu)), Variable(data_h.cuda(gpu)), Variable(target.cuda(gpu))
                output = model(data_l.float(), data_r.float(), data_h.float())
                loss += test_loss(output, target.float()).data
            final_loss += loss
            length += len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}\n'.format(final_loss/length))
    return final_loss/length
