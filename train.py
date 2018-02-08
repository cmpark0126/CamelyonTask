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
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import pylab

from logger import Logger

from load_dataset import get_train_dataset, get_val_dataset

# user define variable
from user_define import Config as cf
from user_define import Hyperparams as hp

import pdb

import random

use_cuda = torch.cuda.is_available()
best_auc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

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
                                          hp.batch_size_for_train,
                                          shuffle=True,
                                          num_workers=32)
valloader = torch.utils.data.DataLoader(valset,
                                        hp.batch_size_for_train,
                                        shuffle=True,
                                        num_workers=32)

# Model
if hp.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoint/ckpt.pth.tar')

    net = checkpoint['net']
    best_auc = checkpoint['AUC']
    start_epoch = checkpoint['epoch']

else:
    print('==> Building model..')
    # net = resnet152()
    # net = densenet121()
    net = inception_v3()

if use_cuda:
    net.cuda()
    range_of_cuda_device = range(torch.cuda.device_count())
    net = torch.nn.DataParallel(net,
                                device_ids=range_of_cuda_device)
    cudnn.benchmark = True


logger = Logger('./logs')

criterion = nn.BCELoss()

optimizer = optim.SGD(net.parameters(), lr=hp.learning_rate,
                      momentum=hp.momentum, weight_decay=hp.weight_decay)
#optimizer = optim.Adam(net.parameters(), lr=hp.learning_rate)
#optimizer = optim.RMSprop(net.parameters(), lr=hp.learning_rate, alpha=0.99)

scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
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

        thresholding = torch.ones(inputs.size(0)) * (1 - hp.threshold_for_train)
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
    global best_auc
    global loss_list

    net.eval()

    val_loss = 0
    correct = 0
    total = 0

    divisor = 200
    section = divisor + 1

    real_tumor = 0
    real_normal= 0

    false_positive = [0] * (section)
    false_negative = [0] * (section)

    sensitivity = []
    specificity = []
    precision = [0] * (section)
    recall = [0] * (section)

    auc = 0
    best_threshold = hp.threshold_for_train
    best_score_inside = 0

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
        

        for i in range(section):
            if i!=0 and i!=divisor:
                thresholding = torch.ones(inputs.size(0)) * (1 - i / divisor)

                predicted = outputs + Variable(thresholding.cuda())
                predicted = torch.floor(predicted)

                find_error = (targets - predicted) * 0.5
                biased = torch.ones(inputs.size(0)) * 0.5

                _false_positive = -find_error + Variable(biased.cuda())
                _false_positive = torch.floor(_false_positive)
                false_positive[i] += _false_positive.data.cpu().sum()

                _false_negative = find_error + Variable(biased.cuda())
                _false_negative = torch.floor(_false_negative)
                false_negative[i] += _false_negative.data.cpu().sum()
        real_tumor += targets.data.cpu().sum()
        real_normal += (hp.batch_size_for_train - targets.data.cpu().sum())

    false_positive[0] = real_normal
    false_negative[divisor] = real_tumor
    for i in range(section):
        if i !=0 and i!= divisor:
            true_positive = real_tumor - false_negative[i]
            precision[i] = true_positive / (true_positive + false_positive[i] + 1e-6)
            recall[i] = true_positive / real_tumor
        elif i == 0:
            precision[i] = 0
            recall[i] = 1
        else:
            precision[i] = 1
            recall[i] = 0
        f_score = 2 * precision[i] * recall[i] / (precision[i] + recall[i]+1e-8)
        if f_score > best_score_inside:
            best_score_inside = f_score
            best_threshold = i

        sensitivity.append(1 - false_negative[i] / real_tumor)
        specificity.append(1 - false_positive[i] / real_normal)
        if i !=0:
            auc += 0.5 * (precision[i] + precision[i-1]) * (-recall[i] + recall[i-1])

        print('Threshold: %.5f | Acc: %.5f%%, Pre: %.5f, Recall: %.5f'
              % (i / divisor,
                100.* (total - false_negative[i] - false_positive[i]) / total,
                precision[i], recall[i]))
    
    plt.plot(recall, precision)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    fig = plt.gcf()
    fig.savefig('ROC_curve.png')
    fig = plt.gcf().clear()
    #============ TensorBoard logging ============#
    acc = 100. * (1 -  (false_negative[best_threshold]+false_positive[best_threshold])/total)
    print('Best score: ', best_score_inside, 'at threshold: ', best_threshold / divisor)
    print('Sensitivity: ', sensitivity[best_threshold], ', Specificity: ', specificity[best_threshold])
    print('Accuracy: ', acc, ', Recall: ', recall[best_threshold], ', Precision: ', precision[best_threshold] )
    print('AUC: ', auc)
    print('FN: ', false_negative[best_threshold], ', FP: ', false_positive[best_threshold], ', RP: ', real_tumor, ', RN: ', real_normal)
    info = {
        'loss': val_loss,
        'Acc': acc,
        'F_score': best_score_inside,
        'AUC': auc
    }
    # (1) Log the scalar values
    for tag, value in info.items():
        logger.scalar_summary(tag, value, epoch + 1)
    
    # (2) Log values and gradients of the parameters (histogram)
    for tag, value in net.named_parameters():
        tag = tag.replace('.', '/')
        logger.histo_summary(tag, to_np(value), epoch + 1)
        logger.histo_summary(tag + '/grad', to_np(value.grad), epoch + 1)
    # Save checkpoint.
    if best_auc < auc:
        print('Saving..')
        state = {
            'net': net.module if use_cuda else net,
            'AUC': auc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/ckpt.pth.tar')
        best_auc = auc
    print(best_auc, ", AUC")

for epoch in range(start_epoch, start_epoch + 10):
#    scheduler.step()
    train(epoch)
    val(epoch)
