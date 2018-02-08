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

# 304
stride = 304
slide_fn = "b_13"

#q = Queue()
#patch_q = Queue()

def makecsv(output, label, size):
    #print(output)
    for i in range(size):
        if output[i] == 1:
            print(label[i])
        wr.writerow([label[i][0], label[i][1], output[i]])

def determine_is_background(patch):
    result_of_sum = np.sum(patch)

    # print(result_of_sum)

    if result_of_sum == 0:
        return True
    elif result_of_sum == 304 * 304:
        return True
    else:
        return False

def make_patch_multi_process(args):
    #print("in make patch")
    #print("in function")
    q, patch_q, pos = args

    target_path = os.path.join(cf.path_of_slide, slide_fn + ".tif")
    slide = openslide.OpenSlide(target_path)

    patch = slide.read_region(pos, 0, hp.patch_size).convert("RGB")
    _div_patch = np.array(patch)
    div_patch = np.divide(_div_patch, 255)

    #if not determine_is_background(div_patch):
    #    return True

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

def _get_interest_region(slide, level, o_knl=5, c_knl=9):
    col, row = slide.level_dimensions[level]

    ori_img = np.array(slide.read_region((0, 0), level, (col, row)))
    img = cv2.cvtColor(ori_img, cv2.COLOR_RGBA2RGB)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    img = img[:,:,1]

    ret, thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    open_knl = np.ones((o_knl, o_knl), dtype = np.uint8)
    close_knl = np.ones((c_knl, c_knl), dtype = np.uint8)

    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, open_knl)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, close_knl)

    _ , contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    xmax = 0
    ymax = 0
    xmin = sys.maxsize
    ymin = sys.maxsize

    print("in makeRECT")
    for i in contours:
        x,y,w,h = cv2.boundingRect(i)
        if(x > xmax):
            xmax = x
        elif(x < xmin):
            xmin = x

        if(y > ymax):
            ymax = y
        elif(y < ymin):
            ymin = y

    downsamples = int(slide.level_downsamples[level])

    return (xmin * downsamples, ymin * downsamples, xmax * downsamples, ymax * downsamples)

if __name__ == "__main__":
    # mp.set_start_method('spawn')
    start_time = time.time()
    use_cuda = torch.cuda.is_available()

    threshold = 0.1
    #batch_size = 200

    f = open(cf.path_for_generated_image + "/" + slide_fn + "_result.csv",
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
    # net.share_memory()
    target_path = os.path.join(cf.path_of_slide, slide_fn + ".tif")
    slide = openslide.OpenSlide(target_path)

    manager = Manager()

    q = manager.Queue()
    patch_q = manager.Queue()

    p = Process(target=manage_q, args=(patch_q, q,))
    p.start()

    p2 = Process(target=manage_q, args=(patch_q, q,))
    p2.start()

    x_min, y_min, x_max, y_max = _get_interest_region(slide, cf.level_for_preprocessing)

    pos = [(x , y) for x in range(x_min, x_max, stride) for y in range(y_min, y_max, stride)]

    #pos.append((-1, -1))

    print("go to map")

    pool = Pool(8)
    result = pool.map_async(make_patch_multi_process, zip(repeat(q), repeat(patch_q), pos))
    # print(result.successful())

    if use_cuda:
        net.cuda()
        net = torch.nn.DataParallel(
        net, device_ids=range(torch.cuda.device_count()))
        cudnn.benchmark = True

    while True :
        if q.qsize() > 0:
            break

    net.eval()
    idx = 0

    print("start")
    while True:
        if q.empty() and patch_q.qsize() < hp.batch_size_for_eval:
            p.terminate()
            p2.terminate()
            print("end loop")
            break

        data = q.get()
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
        print("Queue size is ", q.qsize())
        idx += 1

    print("test loop end")

    p.join()
    p2.join()
    result.wait()
    f.close()
    end_time = time.time()
    print("Program end, Running time is :  ", end_time - start_time)
