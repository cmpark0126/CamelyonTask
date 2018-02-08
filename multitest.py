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
import os
import sys
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

from torch.multiprocessing import Queue, Pool, Process, Manager
from functools import partial

# import tqdm
import time
import pdb

from itertools import repeat
from operator import methodcaller

import cv2

from create_pos_for_test2 import *

# 304

def makecsv(output, label, size):
    #print(output)
    for i in range(size):
        if output[i] == 1:
            print(label[i])
        wr.writerow([label[i][0], label[i][1], output[i]])


def make_patch_multi_process(args):
    #print("in make patch")
    #print("in function")
    manager_q, patch_q, pos = args

    slide_fn = 't_4'

    target_path = os.path.join(cf.path_of_task_2, slide_fn + ".tif")
    slide = openslide.OpenSlide(target_path)

    patch = slide.read_region(pos, 0, hp.patch_size).convert("RGB")
    _div_patch = np.array(patch)
    div_patch = np.divide(_div_patch, 255)

    test_dataset = {}
    test_dataset[cf.key_of_data] = div_patch
    test_dataset[cf.key_of_informs] = np.array(pos)
    patch_q.put(test_dataset)

    #print(patch_q.qsize())
    #print("get a patch")

    slide.close()

def manage_q(patch_q, q):
    while True:
        if patch_q.qsize() >= hp.batch_size_for_eval:
            print("innet")
            test_dataset = {}
            set_of_patch = []
            set_of_pos = []
            for i in range(hp.batch_size_for_eval):
                data = patch_q.get()
                set_of_patch.append(data[cf.key_of_data])
                set_of_pos.append(data[cf.key_of_informs])

            arr = np.array(set_of_patch)
            tset = torch.from_numpy(arr.transpose((0, 3, 1, 2)))

            test_dataset[cf.key_of_data] = tset
            test_dataset[cf.key_of_informs] = np.array(set_of_pos)
            q.put(test_dataset)

def _run(manager_q, patch_q, net, pos_len):
    #pos.append((-1, -1))
    # print(result.successful())

    if use_cuda:
        net.cuda()
        #net = torch.nn.DataParallel(
        #net, device_ids=range(torch.cuda.device_count()))
        #cudnn.benchmark = True

    net.eval()
    idx = 0
    threshold = hp.threshold_for_eval

    while True:
        if idx >= pos_len/hp.batch_size_for_eval:
            print("get out loop")
            break

        data = manager_q.get()
        print("in")

        inputs = data[cf.key_of_data]
        label = data[cf.key_of_informs]

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

        #print(len(outputs_cpu.shape))
        makecsv(outputs_cpu, label, inputs.size(0))
        print("\ntest loop ", idx)
        print("Patch Queue size is ", patch_q.qsize())
        print("Queue size is ", manager_q.qsize())
        idx += 1

    print("test loop end")


if __name__ == "__main__":
    slide_fn = "t_4"
    num_of_poecess = 3

    start_time = time.time()
    use_cuda = torch.cuda.is_available()
    #threshold = 0.1
    #batch_size = 200

    f = open(cf.path_for_generated_image + "/" + slide_fn + "_result.csv",
             'w', encoding='utf-8', newline='')
    wr = csv.writer(f)

    print('==>Resuming from checkpoint..')
    checkpoint = torch.load('./checkpoint/ckpt.pth.tar')
    net = checkpoint['net']
    # net.share_memory()
    target_path = os.path.join(cf.path_of_task_2, slide_fn + ".tif")
    slide = openslide.OpenSlide(target_path)
    level = cf.level_for_preprocessing
    downsamples = int(slide.level_downsamples[level])


    tissue_mask = create_tissue_mask(slide)

    x_min, y_min, x_max, y_max = get_interest_region(tissue_mask)

    stride = cf.stride_for_heatmap
    stride_rescale = int(stride / downsamples)

    set_of_pos = [(x , y) for x in range(x_min, x_max, stride_rescale) for y in range(y_min, y_max, stride_rescale)]

    set_of_real_pos = get_pos_of_patch_for_eval(tissue_mask, set_of_pos)

    set_of_real_pos = np.array(set_of_real_pos)

    col, row = slide.level_dimensions[level]
    thumbnail = slide.get_thumbnail((col, row))
    thumbnail = np.array(thumbnail)

    cv2.imwrite("thumbnail.jpg", thumbnail)

    draw_patch_pos_on_thumbnail(set_of_real_pos, thumbnail, downsamples, slide_fn)

    manager = Manager()
    manager_q = manager.Queue()
    patch_q = manager.Queue()

    p = Process(target=manage_q, args=(patch_q, manager_q,))
    p.start()
#    p2 = Process(target=manage_q, args=(patch_q, q,))
#    p2.start()

    pool = Pool(num_of_poecess)
    result = pool.map_async(make_patch_multi_process, zip(repeat(manager_q), repeat(patch_q), set_of_real_pos))

    _run(manager_q, patch_q, net, len(set_of_real_pos))

    p.join()
    p.terminate()
#    p2.join()
    result.wait()
    f.close()

    end_time = time.time()
    print("Program end, Running time is :  ", end_time - start_time)
