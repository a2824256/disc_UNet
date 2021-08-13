import sys
# sys.path.append('/home/aistudio/external-libraries')
import os
import cv2
import random
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import euclidean_distances
import matplotlib.pylab as plt

import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddle.io import Dataset

images_file = ''  # the path to the training data
gt_file = 'Disc_Cup_Mask/'
test_file = ''  # the path to the testing data
image_size = 256 # the image size to the network (image_size, image_size, 3)
val_ratio = 0.2  # the ratio of train/validation splitition
BATCH_SIZE = 8 # batch size
iters = 3000 # training iteration
optimizer_type = 'adam' # the optimizer, can be set as SGD, RMSprop,...
num_workers = 4 # Number of workers used to load data
init_lr = 1e-3 # the initial learning rate

class FundusDataset(Dataset):
    def __init__(self, image_file, gt_path=None, filelists=None, mode='train'):
        super(FundusDataset, self).__init__()
        self.mode = mode
        self.image_path = image_file
        image_idxs = os.listdir(self.image_path)  # 0001, fundus_img in the folder 0001
        self.gt_path = gt_path

        self.file_list = [image_idxs[i] for i in range(len(image_idxs))]

        if filelists is not None:
            self.file_list = [item for item in self.file_list if item in filelists]

    def __getitem__(self, idx):
        real_index = self.file_list[idx]
        fundus_img_path = os.path.join(self.image_path, real_index, real_index + '.jpg')
        fundus_img = cv2.imread(fundus_img_path)[:, :, ::-1]  # BGR -> RGB
        h, w, c = fundus_img.shape

        if self.mode == 'train':
            gt_tmp_path = os.path.join(self.gt_path, real_index + '.png')
            gt_img = cv2.imread(gt_tmp_path)

            gt_img[gt_img == 128] = 1
            gt_img[gt_img == 255] = 2
            gt_img = cv2.resize(gt_img, (image_size, image_size))
            gt_img = gt_img[:, :, 1]

        fundus_re = cv2.resize(fundus_img, (image_size, image_size))
        img = fundus_re.transpose(2, 0, 1)  # H, W, C -> C, H, W
        # print(img.shape)
        # img = fundus_re.astype(np.float32)

        if self.mode == 'test':
            return img, real_index, h, w

        if self.mode == 'train':
            return img, gt_img

    def __len__(self):
        return len(self.file_list)