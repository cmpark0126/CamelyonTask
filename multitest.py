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

import tqdm
import time
import pdb

from itertools import repeat
from operator import methodcaller

# 304
stride = 304
slide_fn = "b_13"

q = Queue()
patch_q = Queue()

def makecsv(output, label, size):
    #print(output)
    for i in range(size):
        if output[i] == 1:
            print(label[i])
        wr.writerow([label[i][0], label[i][1], output[i]])


"""
def make_patch_process(q):
    target_path = os.path.join(cf.path_of_task_1, slide_fn + ".tif")
    slide = openslide.OpenSlide(target_path)

    set_of_patch = []
    set_of_pos = []

    i = 0
    pbar_total = round(
        ((slide.dimensions[1] / stride) * (slide.dimensions[0] / stride)) / batch_size)
    pbar = tqdm.tqdm(total=pbar_total)
    for y in range(round(slide.dimensions[1] / stride)):
        y *= stride
        for x in range(round(slide.dimensions[0] / stride)):
            x *= stride
            patch = slide.read_region((x, y), 0, hp.patch_size).convert("RGB")
            #img = torch.from_numpy(np.array(patch).transpose((2, 0, 1)))
            # set_of_patch.append(img.float().div(255))
            div_patch = np.array(patch)
            set_of_patch.append(np.divide(div_patch, 255))
            set_of_pos.append(np.array([x, y]))
            i += 1
            if i == batch_size:
                test_dataset = {}
                i = 0
                #np.moveaxis(arr,-1, 0)
                #arr = arr/255
                arr = np.array(set_of_patch)
                tset = torch.from_numpy(arr.transpose((0, 3, 1, 2)))
                # tset.float().div(255)
                # print(arr.shape)
                test_dataset[cf.key_of_data] = tset
                test_dataset[cf.key_of_informs] = np.array(set_of_pos)
                q.put(test_dataset)
                # print(test_dataset[cf.key_of_data].shape)
                # print(test_dataset[cf.key_of_informs].shape)
                set_of_pos.clear()
                set_of_patch.clear()
                pbar.update()
                # print(test_dataset)

    q.put("DONE")
    pbar.close()
"""


def make_patch_multi_process(args):
    #print("in make patch")
    #print("in function")
    q, patch_q, pos = args

    target_path = os.path.join(cf.path_of_task_1, slide_fn + ".tif")
    slide = openslide.OpenSlide(target_path)

    #pos = pos_list

    if pos == (-1, -1):
        print("end queue")
        q.put("DONE")
        return

    patch = slide.read_region(pos, 0, hp.patch_size).convert("RGB")
    div_patch = np.array(patch)
    test_dataset = {}
    test_dataset[cf.key_of_data] = np.divide(div_patch, 255)
    test_dataset[cf.key_of_informs] = np.array(pos)
    patch_q.put(test_dataset)

    #print(patch_q.qsize())
    #print("get a patch")

    # slide.close()

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


if __name__ == "__main__":
    # mp.set_start_method('spawn')
    start_time = time.time()
    use_cuda = torch.cuda.is_available()

    threshold = 0.1
    batch_size = 200

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

    target_path = os.path.join(cf.path_of_task_1, slide_fn + ".tif")
    slide = openslide.OpenSlide(target_path)

    manager = Manager()

    q = manager.Queue()
    patch_q = manager.Queue()

    p = Process(target=manage_q, args=(patch_q, q,))
    p.start()

    pos= [ (x * stride, y * stride) for y in range(round(slide.dimensions[1] / stride)) for x in range(round(slide.dimensions[0] / stride))]
    pos.append((-1, -1))

    print("go to map")

    pool = Pool(4)
    result = pool.map_async(make_patch_multi_process, zip(repeat(q), repeat(patch_q), pos))
    # print(result.successful())

    if use_cuda:
        net.cuda()
        net = torch.nn.DataParallel(
        net, device_ids=range(torch.cuda.device_count()))
        cudnn.benchmark = True

    net.eval()
    idx = 0
    print("start")
    while True:
        data = q.get()
        print("in")
        if data == 'DONE':
            p.terminate()
            break

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

        outputs_cpu = [0]*inputs.size(0)

        #print(len(outputs_cpu.shape))
        makecsv(outputs_cpu, label, inputs.size(0))
        print("\ntest loop ", idx)
        print("Patch Queue size is ", patch_q.qsize())
        idx += 1

    print("test loop end")

    p.join()
    result.wait()
    f.close()
    end_time = time.time()
    print("Program end, Running time is :  ", end_time - start_time)
