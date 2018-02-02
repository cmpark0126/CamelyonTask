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

from load_dataset import get_dataset

import csv
from user_define import Config as cp

vuse_cuda = torch.cuda.is_available()

threshold = 0.7
batch_size = 100
tumor_list = []
labeling = []

def makecsv(output, label):
    f = open(cp.path_for_generated_image + "/result.csv", 'w', encoding = 'utf-8', newline='')
    wr = csv.writer(f)
    for i in range(batch_size):
        print(label[i])
        wr.writerow([label[i], output[i]])
    f.close()

print('==> Preparing data..')
transform_test =transforms.Compose([
    transforms.ToTensor(),
])

trainset, valset, testset = get_dataset(transform_test, transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size, shuffle=False, num_workers=16)

print('==>Resuming from checkpoint..')
checkpoint = torch.load('./checkpoint/ckpt.t7')
net = checkpoint['net']



if use_cuda:
    net.cuda()
    net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))
    cudnn.benchmark = True


net.eval()

for batch_idx, (inputs, label ) in enumerate(testloader):
    if use_cuda:
        inputs = inputs.type(torch.cuda.FloatTensor)
        inputs = inputs.cuda()

    inputs = Variable(inputs, volatile=True)
    outputs = net(inputs)
    outputs = torch.squeeze(outputs)
    thresholding = torch.ones(batch_size) * (1 - threshold)
    outputs = outputs + Variable(thresholding.cuda())
    outputs = torch.floor(outputs)
    outputs_cpu = outputs.cpu()
    makecsv(output, label)

print("end")
