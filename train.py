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

from load_dataset import get_dataset

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.01, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
args = parser.parse_args()

use_cuda = torch.cuda.is_available()
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

threshold = 0.7
batch_size = 10




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



trainset, valset, testset = get_dataset(transform_train, transform_test)
trainloader = torch.utils.data.DataLoader(trainset, batch_size, shuffle=True, num_workers=16)
valloader = torch.utils.data.DataLoader(valset, batch_size, shuffle=False, num_workers=16)
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
    net = resnet101()
    #net = DenseNet121()

if use_cuda:
    net.cuda()
    net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))
    cudnn.benchmark = True

criterion = nn.BCELoss()
optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=9e-4)
#optimizer = optim.Adam(net.parameters(), lr=args.lr)
#optimizer = optim.RMSprop(net.parameters(), lr=args.lr, alpha=0.99)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)

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
#        pdb.set_trace()
        outputs = torch.squeeze(outputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        thresholding = torch.ones(batch_size) * (1 - threshold)
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

    val_loss = 0
    correct = 0
    total = 0

    for batch_idx, (inputs, targets) in enumerate(valloader):
        if use_cuda:
            inputs, targets = inputs.type(torch.cuda.FloatTensor), targets.type(torch.cuda.FloatTensor)
            inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = Variable(inputs, volatile=True), Variable(targets)
        outputs = net(inputs)
        outputs = torch.squeeze(outputs)
        loss = criterion(outputs, targets)
        thresholding = torch.ones(batch_size) * (1 - threshold)
        predicted = outputs + Variable(thresholding.cuda())
        predicted = torch.floor(predicted)

        val_loss += loss.data[0]
        total += targets.size(0)
        correct += predicted.data.eq(targets.data).cpu().sum()

        progress_bar(batch_idx, len(valloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            % (val_loss/(batch_idx+1), 100.*correct/total, correct, total))

    # Save checkpoint.
    acc = 100.*correct/total
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





def test():
    global best_acc
    test_loss = 0
    correct = 0
    total = 0
    checkpoint = torch.load('./checkpoint/ckpt.t7')
    net = checkpoint['net']
    epoch = checkpoint['epoch']
    adjust_learning_rate(optimizer, epoch)

    net.eval()

    for batch_idx, (inputs, targets) in enumerate(testloader):
        if use_cuda:
            inputs, targets = inputs.type(torch.cuda.FloatTensor), targets.type(torch.cuda.FloatTensor)
            inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = Variable(inputs, volatile=True), Variable(targets)
        outputs = net(inputs)
        outputs = torch.squeeze(outputs)
        loss = criterion(outputs, targets)
        thresholding = torch.ones(batch_size) * (1 - threshold)
        predicted = outputs + Variable(thresholding.cuda())
        predicted = torch.floor(predicted)

        test_loss += loss.data[0]
        total += targets.size(0)
        correct += outputs.data.eq(targets.data).cpu().sum()

        progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))



for epoch in range(start_epoch, start_epoch+1):
    scheduler.step()
    train(epoch)
    val(epoch)

# test()
