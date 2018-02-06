from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets

import os
# import argparse

from models import *
from utils import progress_bar
from torch.autograd import Variable

import numpy as np
import pylab
# import matplotlib
# matplotlib.use('agg')
# import matplotlib.pyplot as plt

from logger import Logger

from load_dataset import get_train_dataset, get_val_dataset

# user define variable
from user_define import Config as cf
from user_define import Hyperparams as hp

import pdb

use_cuda = torch.cuda.is_available()
best_score = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

threshold = 0.2

def to_np(x):
    return x.data.cpu().numpy()

# Data
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(180),
    transforms.RandomGrayscale(p=0.1),
    transforms.ToTensor()
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
])

trainset = get_train_dataset(transform_train, transform_test)
valset = get_val_dataset(transform_train, transform_test)

trainloader = torch.utils.data.DataLoader(trainset,
                                          hp.batch_size,
                                          shuffle=True,
                                          num_workers=32)
valloader = torch.utils.data.DataLoader(valset,
                                        hp.batch_size,
                                        shuffle=True,
                                        num_workers=32)

# Model
if hp.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoint/ckpt.pth.tar')

    net = checkpoint['net']
    best_score = checkpoint['score']
    start_epoch = checkpoint['epoch']

else:
    print('==> Building model..')
    net = resnet152()
    # net = densenet121()
    # net = inception_v3()

if use_cuda:
    net.cuda()
    net = torch.nn.DataParallel(net,
                                device_ids=range(torch.cuda.device_count()))
    cudnn.benchmark = True


logger = Logger('./logs')

criterion = nn.BCELoss()

optimizer = optim.SGD(net.parameters(), lr=hp.learning_rate,
                      momentum=hp.momentum, weight_decay=hp.weight_decay)
#optimizer = optim.Adam(net.parameters(), lr=hp.learning_rate)
#optimizer = optim.RMSprop(net.parameters(), lr=hp.learning_rate, alpha=0.99)

scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
#scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3, verbose=True, threshold = 0.001)


# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)

    net.train()

    train_loss = 0
    correct = 0
    total = 0

    for batch_idx, (inputs, targets) in enumerate(trainloader):
        if use_cuda:
            inputs = inputs.type(torch.cuda.FloatTensor)
            targets = targets.type(torch.cuda.FloatTensor)

            inputs, targets = inputs.cuda(), targets.cuda()

        optimizer.zero_grad()
        inputs, targets = Variable(inputs), Variable(targets)

        outputs = net(inputs)
        outputs = torch.squeeze(outputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        thresholding = torch.ones(inputs.size(0)) * (1 - threshold)
        predicted = outputs + Variable(thresholding.cuda())
        predicted = torch.floor(predicted)

        train_loss += loss.data[0]
        total += targets.size(0)
        correct += predicted.data.eq(targets.data).cpu().sum()

        progress_bar(batch_idx,
                     len(trainloader),
                     'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (train_loss / (batch_idx + 1),
                        100. * correct / total,
                        correct, total))


def val(epoch):
    global best_score
    global loss_list

    net.eval()

    hubo_num = 50
    val_loss = 0
    correct = 0
    total = 0

    real_tumor = 0
    real_normal= 0

    false_positive = [0] * (hubo_num + 1)
    false_negative = [0] * (hubo_num + 1)

    sensitivity = []
    specificity = []

    best_score_inside = 0
    best_threshold = 0.2
    best_recall = 0
    best_precision = 0

    for batch_idx, (inputs, targets) in enumerate(valloader):
        if use_cuda:
            inputs = inputs.type(torch.cuda.FloatTensor)
            targets = targets.type(torch.cuda.FloatTensor)

            inputs, targets = inputs.cuda(), targets.cuda()

        inputs, targets = Variable(inputs, volatile=True), Variable(targets)

        outputs = net(inputs)
        outputs = torch.squeeze(outputs)

        loss = criterion(outputs, targets)
        val_loss += loss.data[0]
        total += targets.size(0)

        for i in range(hubo_num + 1):
            thresholding = torch.ones(inputs.size(0)) * (1 - i / hubo_num)

            predicted = outputs + Variable(thresholding.cuda())
            predicted = torch.floor(predicted)

            finderror = (targets - predicted) * 0.5
            biased = torch.ones(inputs.size(0)) * 0.5

            fposi = finderror + Variable(biased.cuda())
            fnega = -finderror + Variable(biased.cuda())
            fposi = torch.floor(fposi)
            fnega = torch.floor(fnega)

            false_positive[i] += fposi.data.cpu().sum()
            false_negative[i] += fnega.data.cpu().sum()

        real_tumor += targets.data.cpu().sum()
        real_normal += (hp.batch_size - targets.data.cpu().sum())

    for i in range(hubo_num + 1):
        true_positive = real_tumor - false_negative[i]

        precision = true_positive / (true_positive + false_positive[i])
        recall = true_positive / (true_positive + false_negative[i])

        f_score = 2 * precision * recall / (precision + recall)

        if f_score > best_score_inside:
            best_score_inside = f_score
            best_threshold = i
            best_recall = recall
            best_precision = precision

        sensitivity.append(1 - false_negative[i] / real_tumor)
        specificity.append(1 - false_positive[i] / real_normal)

        print('Threshold: %.5f | Acc: %.5f%%'
              % (i / hubo_num,
                 (total - false_negative[i] - false_positive[i]) / total))
        # print("Threshold: ", (i / hubo_num),
        #       ", Accuracy: ", (total - false_negative[i] - false_positive[i]) / total)

    # for save fig
    plt.plot(specificity, sensitivity)
    plt.xlabel('Specificity')
    plt.ylabel('Sensitivity')
    fig = plt.gcf()
    fig.savefig('ROC_curve.png')
    fig = plt.gcf().clear()


# Save checkpoint.

    acc = 100. * (1 -  (false_negative[best_threshold]+false_positive[best_threshold])/total)
    print('Best score: ', best_score_inside, 'at threshold: ', best_threshold / hubo_num)
    print('Sensitivity: ', sensitivity[best_threshold], ', Specificity: ', specificity[best_threshold])
    print('Accuracy: ', acc, ', Recall: ', best_recall, ', Precision: ', best_precision )
    info = {
        'loss': val_loss,
        'Acc': acc,
        'F_score': best_score_inside
    }

    #============ TensorBoard logging ============#
    # (1) Log the scalar values

    for tag, value in info.items():
        logger.scalar_summary(tag, value, epoch + 1)

    # (2) Log values and gradients of the parameters (histogram)
    for tag, value in net.named_parameters():
        tag = tag.replace('.', '/')
        logger.histo_summary(tag, to_np(value), epoch + 1)
        logger.histo_summary(tag + '/grad', to_np(value.grad), epoch + 1)

    # (3) Log the images
#       info = {
#           'images': to_np(images.view(-1, 28, 28)[:10])
#       }

#       for tag, images in info.items():
#           logger.image_summary(tag, images, step+1)
    if best_score < best_score_inside:
        print('Saving..')
        state = {
            'net': net.module if use_cuda else net,
            'score': best_score_inside,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/ckpt.pth.tar')
        best_score = best_score_inside
    print(best_score, ", F-score")



for epoch in range(start_epoch, start_epoch + 15):
    scheduler.step()
    train(epoch)
    val(epoch)
