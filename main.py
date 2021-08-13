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
from model import cup_disc_UNet
from util import DiceLoss

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

filelists = os.listdir(images_file)
train_filelists, val_filelists = train_test_split(filelists, test_size = val_ratio,random_state = 42)
print("Total Nums: {}, train: {}, val: {}".format(len(filelists), len(train_filelists), len(val_filelists)))

_train = FundusDataset(image_file = images_file,
                        gt_path = gt_file)

_val = FundusDataset(image_file = images_file,
                        gt_path = gt_file)


def train(model, iters, train_dataloader, val_dataloader, optimizer, criterion, metric, log_interval, evl_interval):
    iter = 0
    model.train()
    avg_loss_list = []
    avg_dice_list = []
    best_dice = 0.
    while iter < iters:
        for data in train_dataloader:
            iter += 1
            if iter > iters:
                break
            fundus_img = (data[0]/255.).astype("float32")
            gt_label = (data[1]).astype("int64")
            # print('label shape: ', gt_label.shape)
            logits = model(fundus_img)
            # print('logits shape: ', logits.shape)
            loss = criterion(logits, gt_label)
            # print('loss: ',loss)
            dice = metric(logits, gt_label)
            # print('dice: ', dice)

            loss.backward()
            optimizer.step()

            model.clear_gradients()
            avg_loss_list.append(loss.numpy()[0])
            avg_dice_list.append(dice.numpy()[0])

            if iter % log_interval == 0:
                avg_loss = np.array(avg_loss_list).mean()
                avg_dice = np.array(avg_dice_list).mean()
                avg_loss_list = []
                avg_dice_list = []
                print("[TRAIN] iter={}/{} avg_loss={:.4f} avg_dice={:.4f}".format(iter, iters, avg_loss, avg_dice))

            if iter % evl_interval == 0:
                avg_loss, avg_dice = val(model, val_dataloader)
                print("[EVAL] iter={}/{} avg_loss={:.4f} dice={:.4f}".format(iter, iters, avg_loss, avg_dice))
                if avg_dice >= best_dice:
                    best_dice = avg_dice
                    paddle.save(model.state_dict(),
                                os.path.join("best_model_{:.4f}".format(best_dice), 'model.pdparams'))
                model.train()


def val(model, val_dataloader):
    model.eval()
    avg_loss_list = []
    avg_dice_list = []
    with paddle.no_grad():
        for data in val_dataloader:
            fundus_img = (data[0] / 255.).astype("float32")
            gt_label = (data[1]).astype("int64")

            pred = model(fundus_img)
            loss = criterion(pred, gt_label)
            dice = metric(pred, gt_label)

            avg_loss_list.append(loss.numpy()[0])
            avg_dice_list.append(dice.numpy()[0])

    avg_loss = np.array(avg_loss_list).mean()
    avg_dice = np.array(avg_dice_list).mean()

    return avg_loss, avg_dice

train_dataset = FundusDataset(image_file = images_file,
                        gt_path = gt_file,
                        filelists=train_filelists)

val_dataset = FundusDataset(image_file = images_file,
                        gt_path = gt_file,
                        filelists=val_filelists)

train_loader = paddle.io.DataLoader(
    train_dataset,
    batch_sampler=paddle.io.DistributedBatchSampler(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=False),
    num_workers=num_workers,
    return_list=True,
    use_shared_memory=False
)

val_loader = paddle.io.DataLoader(
    val_dataset,
    batch_sampler=paddle.io.DistributedBatchSampler(val_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=False),
    num_workers=num_workers,
    return_list=True,
    use_shared_memory=False
)

model = cup_disc_UNet(num_classes=3)

### The SUMMARY interface provided by the paddlepaddle is called to visualize the constructed model,
### which is convenient to view and confirm the model structure and parameter information.
# paddle.Model(model).summary((-1,3,256,256))

if optimizer_type == "adam":
    optimizer = paddle.optimizer.Adam(init_lr, parameters=model.parameters())

criterion = nn.CrossEntropyLoss(axis=1)
metric = DiceLoss()
train(model, iters, train_loader, val_loader, optimizer, criterion, metric, log_interval=50, evl_interval=100)
