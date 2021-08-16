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
from dataset import FundusDataset
from paddleseg.models import AttentionUNet
from model import cup_disc_UNet
from util import DiceLoss
import warnings
warnings.filterwarnings('ignore')
test_file = 'E:\\disc_UNet\\multi-modality_images'

best_model_path = "./best_model_0.8765/model.pdparams"
model = AttentionUNet(num_classes=3)
para_state_dict = paddle.load(best_model_path)
model.set_state_dict(para_state_dict)
model.eval()

test_dataset = FundusDataset(image_file=test_file,
                            mode='test')

for fundus_img, idx, h, w in test_dataset:
    # print(idx)
    fundus_img = fundus_img[np.newaxis, ...]
    fundus_img = paddle.to_tensor((fundus_img / 255.).astype("float32"))
    logits = model(fundus_img)
    pred_img = logits.numpy().argmax(1)
    pred_gray = np.squeeze(pred_img, axis=0)
    pred_gray = pred_gray.astype('float32')
    # print(pred_gray.shape)
    pred_gray[pred_gray == 1] = 128
    pred_gray[pred_gray == 2] = 255
    # print(pred_gray)
    pred_ = cv2.resize(pred_gray, (w, h))
    # print(pred_.shape)
    cv2.imwrite('./Disc_Cup_Segmentations/'+idx+'.bmp', pred_)

