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

from load_dataset import get_test_dataset
from load_dataset import get_val_dataset

import csv

# user define variable
from user_define import Config as cf
from user_define import Hyperparams as hp

use_cuda = torch.cuda.is_available()

print('==> Preparing data..')
transform_test = transforms.Compose([
    transforms.ToTensor(),
])
testset = get_test_dataset(transform_test, transform_test)
testloader = torch.utils.data.DataLoader(testset,
                                         hp.batch_size_for_eval,
                                         shuffle=False,
                                         num_workers=16)

print('==> Resuming from checkpoint..')
checkpoint = torch.load('./checkpoint/ckpt.pth.tar')
net = checkpoint['net']

if use_cuda:
    net.cuda()
    net = torch.nn.DataParallel(
        net, device_ids=range(torch.cuda.device_count()))
    cudnn.benchmark = True

def makecsv(file_writer, output, label, size):
    for i in range(size):
        file_writer.writerow([label[i], output[i]])

def eval_for_task1():
    net.eval()
    fn = os.path.join(cf.path_for_generated_image, 'result.csv')
    fo = open(fn, 'w', encoding='utf-8', newline='')
    fw = csv.writer(fo)

    for batch_idx, (inputs, label) in enumerate(testloader):
        if use_cuda:
            inputs = inputs.type(torch.cuda.FloatTensor)
            inputs = inputs.cuda()

        inputs = Variable(inputs, volatile=True)

        outputs = net(inputs)
        outputs = torch.squeeze(outputs)
        thresholding = torch.ones(inputs.size(0)) * (1 - hp.threshold_for_eval)
        outputs = outputs + Variable(thresholding.cuda())
        outputs = torch.floor(outputs)
        outputs_cpu = outputs.data.cpu()

        makecsv(fw, outputs_cpu, label, inputs.size(0))

    fo.close()

eval_for_task1()

print("end")
