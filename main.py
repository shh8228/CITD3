import argparse

import sys
import time
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import numpy as np
import torch.distributed as dist
import torch
from torchvision import models

from train import test, train
from network import GazeEstimationAbstractModel, GazeEstimationModelVGG

sys.path.append('/home/dicetemp/CITD3/apex/apex')
from apex.parallel import DistributedDataParallel as DDP

def parse():
    model_names = sorted(name for name in models.__dict__
                         if name.islower() and not name.startswith("__")
                         and callable(models.__dict__[name]))

    parser = argparse.ArgumentParser(description='CITD3')
    parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet18',
                        choices=model_names,
                        help='model architecture: ' +
                             ' | '.join(model_names) +
                             ' (default: resnet18)')
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--epochs', default=90, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('-b', '--batch-size', default=256, type=int,
                        metavar='N', help='mini-batch size per process (default: 256)')
    parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                        metavar='LR',
                        help='Initial learning rate.  Will be scaled by <global batch size>/256: args.lr = args.lr*float(args.batch_size*args.world_size)/256.  A warmup schedule will also be applied over the first 5 epochs.')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)')
    parser.add_argument('--print-freq', '-p', default=10, type=int,
                        metavar='N', help='print frequency (default: 10)')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                        help='evaluate model on validation set')
    parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                        help='use pre-trained model')

    parser.add_argument('--prof', default=-1, type=int,
                        help='Only run 10 iterations for profiling.')
    parser.add_argument('--deterministic', action='store_true')

    parser.add_argument("--local_rank", default=0, type=int)
    parser.add_argument('--sync_bn', action='store_true',
                        help='enabling apex sync BN.')

    parser.add_argument('--opt-level', type=str)
    parser.add_argument('--keep-batchnorm-fp32', type=str, default=None)
    parser.add_argument('--loss-scale', type=str, default=None)
    parser.add_argument('--channels-last', type=bool, default=False)
    args = parser.parse_args()
    return args


def main():
    global args
    
    args = parse()
    args.gpu = 0
    args.world_size = 1

    x = np.arange(0, args.epochs)
    y_v = np.arange(0, args.epochs).astype(np.float64)
    y_t = np.arange(0, args.epochs).astype(np.float64)

    args.gpu = args.local_rank
    torch.cuda.set_device(args.gpu)
    torch.distributed.init_process_group(backend='nccl',
                                         init_method='env://')
    args.world_size = dist.get_world_size()

    model = GazeEstimationModelVGG(num_out=2)
    model.cuda(args.gpu)
    model = DDP(model, delay_allreduce=True)
    model = model.float()

    start_time = time.strftime('%c', time.localtime(time.time()))
    for i in range(args.epochs):
        model, validation_loss  = train(i, model, args.batch_size, args.workers, args.gpu)
        test_loss = test(model, args.batch_size, args.workers, args.gpu)
        y_v[i], y_t[i] = validation_loss, test_loss
    
    end_time = time.strftime('%c', time.localtime(time.time()))

    torch.save(model.state_dict(), './model.pt')

    np.savetxt('validation_loss_'+ str(args.gpu) + '.txt', y_v, fmt='%1.4f')
    np.savetxt('test_loss_'+ str(args.gpu) + '.txt', y_t, fmt='%1.4f')
    np.savetxt('epochs.txt', x, header='--start time : ' + start_time + '--', footer='--end time : ' + end_time + '--', fmt='%d')

    plt.plot(x,y_t)
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.title('test loss')
    plt.savefig('test_graph_'+ str(args.gpu) + '.png')
    
    plt.plot(x,y_v)
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.title('test and validation loss')
    plt.savefig('graphs_'+ str(args.gpu) + '.png')
   
	

if __name__=="__main__":
    main()
