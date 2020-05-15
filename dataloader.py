import torch
import os
import string
from PIL import Image
from torchvision.transforms import ToTensor
import numpy as np
'''
import things
'''

for i in range(0, 17):
    sessStr = 's' + str(i).zfill(3) + '_glasses'

    path = os.path.join('D:/anaconda3/envs/citd3/dataset', sessStr, sessStr)

    for f in ['left', 'right']:
        image_np = np.empty((1, 3, 36, 60), dtype='float32')
        for fname in os.listdir(os.path.join(path, 'inpainted', f)):
            fpath = os.path.join(path, 'inpainted', f, fname)
            image_file = Image.open(fpath)
            image_file = ToTensor()(image_file).unsqueeze(0)
            image_file = image_file.numpy()
            image_np = np.append(image_np, image_file, axis=0)
        image_np = np.delete(image_np, [0, 0], axis=0)
        np.save(os.path.join(path, f + '_x'), image_np)

    fo = open(os.path.join(path, "label_combined.txt"), 'r')
    y_np = np.empty((1, 2), dtype='float32')
    h_np = np.empty((1, 2), dtype='float32')
    for line in fo.readlines():
        np.append(y_np, np.array([[float(line.split(' ')[3][1:-1]), float(line.split(' ')[4][0:-2])]]), axis=0)
        np.append(h_np, np.array([[float(line.split(' ')[1][1:-1]), float(line.split(' ')[2][0:-2])]]), axis=0)
    y_np = np.delete(y_np, [0, 0], axis=0)
    h_np = np.delete(h_np, [0, 0], axis=0)
    np.save(os.path.join(path, 'y_val'), y_np)
    np.save(os.path.join(path, 'headPose_x'), h_np)
    fo.close()
'''
face image not transformed into tensor, but used for opencv&Dlib pose estimator
x_data(head pose) generation  
'''
