from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.multiprocessing as mp

import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import os, sys
import argparse

from models import *
from utils import progress_bar
from torch.autograd import Variable

import openslide

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pylab

from load_dataset import get_test_dataset
from load_dataset import get_val_dataset

import csv
from user_define import Config as cf
from user_define import Hyperparams as hp

from torch.multiprocessing import Queue, Pool, Process

import tqdm
import time
import pdb


def makecsv(output, label, size):
    for i in range(size):
        if output[i] == 1 :
            print(label[i])
        wr.writerow([label[i][0],label[i][1], output[i]])

def make_patch_process(q) :
    target_path = os.path.join(cf.path_of_task_1, 'b_15.tif')
    slide = openslide.OpenSlide(target_path)

    set_of_patch = []
    set_of_pos = []

    i = 0
    pbar_total = round(((slide.dimensions[1]/304)*(slide.dimensions[0]/304))/batch_size)
    pbar = tqdm.tqdm(total = pbar_total)
    for y in range(round(slide.dimensions[1]/304)):
        y *= 304
        for x in range(round(slide.dimensions[0]/304)):
            x *= 304
            patch = slide.read_region((x,y) , 0 ,hp.patch_size).convert("RGB")
            #img = torch.from_numpy(np.array(patch).transpose((2, 0, 1)))
            #set_of_patch.append(img.float().div(255))
            div_patch = np.array(patch)
            set_of_patch.append(np.divide(div_patch, 255))
            set_of_pos.append(np.array([x,y]))
            i += 1
            if i == batch_size:
                test_dataset = {}
                i = 0
                #np.moveaxis(arr,-1, 0)
                #arr = arr/255
                arr = np.array(set_of_patch)
                tset = torch.from_numpy(arr.transpose((0,3,1,2)))
                #tset.float().div(255)
                #print(arr.shape)
                test_dataset[cf.key_of_data] = tset
                test_dataset[cf.key_of_informs] = np.array(set_of_pos)
                q.put(test_dataset)
                #print(test_dataset[cf.key_of_data].shape)
                #print(test_dataset[cf.key_of_informs].shape)
                set_of_pos.clear()
                set_of_patch.clear()
                pbar.update()
                #print(test_dataset)
                #time.sleep()
    q.put("DONE")
    pbar.close()

if __name__ == "__main__":
    #mp.set_start_method('spawn')
    use_cuda = torch.cuda.is_available()

    threshold = 0.1
    batch_size = 250
    tumor_list = []
    labeling = []


    f = open(cf.path_for_generated_image + "/result.csv",
             'w', encoding='utf-8', newline='')
    wr = csv.writer(f)

    print('==> Preparing data..')
    transform_test = transforms.Compose([
        transforms.ToTensor(),
    ])

    print('==>Resuming from checkpoint..')
    checkpoint = torch.load('./checkpoint/ckpt.pth.tar')
    # print(checkpoint)
    net = checkpoint['net']
    net.share_memory()

    q = Queue()
    p = Process(target = make_patch_process, args=(q,))
    p.start()

    if use_cuda:
        net.cuda()
        #net = torch.nn.DataParallel(
                #net, device_ids=range(torch.cuda.device_count()))
        #cudnn.benchmark = True

    net.eval()
    i = 0
    while True:
        test_data= q.get()
        if(test_data == 'DONE'):
            break

        inputs = test_data[cf.key_of_data]
        label = test_data[cf.key_of_informs]
            # print(label)
            # print(label.shape)

        if use_cuda:
            inputs = inputs.type(torch.cuda.FloatTensor)
            inputs = inputs.cuda()
        inputs = Variable(inputs, volatile=True)
        outputs = net(inputs)
        outputs = torch.squeeze(outputs)
        thresholding = torch.ones(inputs.size(0)) * (1 - threshold)
                # print(outputs)
        outputs = outputs + Variable(thresholding.cuda())
        outputs = torch.floor(outputs)
        outputs_cpu = outputs.data.cpu()

        makecsv(outputs_cpu, label, inputs.size(0))
        print("test loop ", i )
        print("Queue size is ", q.qsize())
        i += 1

    print("test loop end")

p.join()
f.close()

print("program end")
