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
import argparse

from models import *
from utils import progress_bar
from torch.autograd import Variable

import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import pylab

import pdb

from logger import Logger

from load_dataset import get_train_dataset, get_val_dataset

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.01, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
args = parser.parse_args()

use_cuda = torch.cuda.is_available()
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

threshold = 0.2
batch_size = 250

def to_np(x):
    return x.data.cpu().numpy()


# Data
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ToTensor()
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
])

trainset = get_train_dataset(transform_train, transform_test)
valset = get_val_dataset(transform_train, transform_test)
trainloader = torch.utils.data.DataLoader(trainset, batch_size, shuffle=True, num_workers=16)
valloader = torch.utils.data.DataLoader(valset, batch_size, shuffle=True, num_workers=16)
# testloader = torch.utils.data.DataLoader(testset, batch_size, shuffle=False, num_workers=2)



# Model
if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoint/ckpt.t7')
    net = checkpoint['net']
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']

else:
    print('==> Building model..')
#    net = resnet101()
    net = densenet121()

if use_cuda:
    net.cuda()
    net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))
    cudnn.benchmark = True


logger = Logger('./logs')
criterion = nn.BCELoss()
optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=9e-4)
#optimizer = optim.Adam(net.parameters(), lr=args.lr)
#optimizer = optim.RMSprop(net.parameters(), lr=args.lr, alpha=0.99)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)

    net.train()
    train_loss = 0
    correct = 0
    total = 0

    for batch_idx, (inputs, targets) in enumerate(trainloader):
        if use_cuda:
            inputs, targets = inputs.type(torch.cuda.FloatTensor), targets.type(torch.cuda.FloatTensor)
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

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))

def val(epoch):
    global best_acc
    global loss_list


    net.eval()

    hubo_num = 20
    val_loss = 0
    correct = 0
    total = 0
    positive = 0
    negative = 0
    false_positive = [0] * (hubo_num + 1)
    false_negative = [0] * (hubo_num + 1)
    sensitivity = []
    specificity = []
    best_correction = 0
    best_threshold = 0.2

    for batch_idx, (inputs, targets) in enumerate(valloader):
        if use_cuda:
            inputs, targets = inputs.type(torch.cuda.FloatTensor), targets.type(torch.cuda.FloatTensor)
            inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = Variable(inputs, volatile=True), Variable(targets)
        outputs = net(inputs)
        outputs = torch.squeeze(outputs)
        loss = criterion(outputs, targets)
        val_loss += loss.data[0]
        total += targets.size(0)
        for i in range(hubo_num+1):
            thresholding = torch.ones(inputs.size(0)) * (1 - i/hubo_num)
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
        positive += targets.data.cpu().sum()
        negative += (batch_size - targets.data.cpu().sum())

    for i in range(hubo_num+1):
        error = false_negative[i] + false_positive[i]
        if total - error < best_correction:
            best_correction = total - error
            best_threshold = i / hubo_num
        sensitivity.append(1 - false_negative[i] / positive)
        specificity.append(1 - false_positive[i] / negative)
    
    plt.plot(specificity, sensitivity)
    plt.xlabel('Specificity')
    plt.ylabel('Sensitivity')
    fig = plt.gcf()
    fig.savefig('ROC curve.png')
    print(sensitivity, ", sensitivity, ", specificity, ", specificity")
    fig = plt.gcf().clear()


# Save checkpoint.

    acc = 100.*best_correction/total
    print('Best accuracy: ', acc, 'at threshold: ', best_threshold )
    info = {
            'loss': loss.data[0],
            'accuracy': 100.*correct/total
             }

    #============ TensorBoard logging ============#
    # (1) Log the scalar values

    for tag, value in info.items():
        logger.scalar_summary(tag, value, epoch+1)

    # (2) Log values and gradients of the parameters (histogram)
    for tag, value in net.named_parameters():
        tag = tag.replace('.', '/')
        logger.histo_summary(tag, to_np(value), epoch+1)
        logger.histo_summary(tag+'/grad', to_np(value.grad), epoch+1)

    # (3) Log the images
#       info = {
#           'images': to_np(images.view(-1, 28, 28)[:10])
#       }

#       for tag, images in info.items():
#           logger.image_summary(tag, images, step+1)


    if acc > best_acc:
        print('Saving..')
        state = {
            'net' : net.module if use_cuda else net,
            'acc' : acc,
            'epoch' : epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/ckpt.t7')
        best_acc = acc



for epoch in range(start_epoch, start_epoch+50):
    scheduler.step()
    train(epoch)
    val(epoch)

