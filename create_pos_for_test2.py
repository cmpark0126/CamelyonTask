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

import time
import pdb

from itertools import repeat
from operator import methodcaller

import cv2

def get_interest_region(tissue_mask, o_knl=5, c_knl=9):
    open_knl = np.ones((o_knl, o_knl), dtype = np.uint8)
    close_knl = np.ones((c_knl, c_knl), dtype = np.uint8)

    tissue_mask = cv2.morphologyEx(tissue_mask, cv2.MORPH_OPEN, open_knl)
    tissue_mask = cv2.morphologyEx(tissue_mask, cv2.MORPH_CLOSE, close_knl)

    _ , contours, hierarchy = cv2.findContours(tissue_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

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

    return xmin, ymin, xmax, ymax

def create_tissue_mask(slide):
    level = cf.level_for_preprocessing

    col, row = slide.level_dimensions[level]

    img = np.array(slide.read_region((0, 0), level, (col, row)))
    img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    img = img[:, :, 1]

    _, tissue_mask = cv2.threshold(img,
                                   0,
                                   255,
                                   cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # target_image_path = os.path.join(self.etc_path,
    #                                  "tissue_mask.jpg")
    # cv2.imwrite(target_image_path, tissue_mask)

    return tissue_mask

def get_pos_of_patch_for_eval(mask, set_of_pos):
    level = cf.level_for_preprocessing
    downsamples = 2 ** level
    gap = int(304 / downsamples)

    print(mask.shape)
    length = len(set_of_pos)

    set_of_real_pos = []

    j = 0

    for i in range(length):
        x, y = set_of_pos[i]
        x_, y_ = (x + gap), (y + gap)
        patch = mask[y:y_, x:x_]
        if determine_is_background(patch):
            continue
        else:
            xreal, yreal = x*downsamples, y*downsamples
            set_of_real_pos.append((xreal, yreal))
            j = j + 1
        print("\r %d/%d correct : %d" % (i, length, j), end="")

    print("\n")
    return set_of_real_pos

def determine_is_background(patch):
    area = patch.size
    _sum = np.sum(patch)

    ratio = _sum / area

    if ratio > cf.ratio_of_tissue_area:
        return False #is not background
    else:
        return True


def draw_patch_pos_on_thumbnail(set_of_real_pos, thumbnail, downsamples, slide_fn):
    for pos in set_of_real_pos:
        x, y = pos
        min_x = int(x/downsamples)
        min_y = int(y/downsamples)
        max_x = min_x + int(304/downsamples)
        max_y = min_y + int(304/downsamples)

        cv2.rectangle(thumbnail,
                      (min_x, min_y),
                      (max_x, max_y),
                      (255, 0, 0),
                      4)
    target_path = os.path.join(cf.path_for_generated_image, "patch_pos_to_thumbnail_" + slide_fn + ".jpg")
    cv2.imwrite(target_path, thumbnail)


if __name__ == "__main__":
    # f = open(cf.path_for_generated_image + "/" + slide_fn + "_result.csv",
    #          'w', encoding='utf-8', newline='')
    # wr = csv.writer(f)

    target_path = os.path.join(cf.path_of_task_1, 'b_2' + ".tif")
    slide = openslide.OpenSlide(target_path)
    level = cf.level_for_preprocessing
    downsamples = int(slide.level_downsamples[level])

    tissue_mask = create_tissue_mask(slide)

    x_min, y_min, x_max, y_max = get_interest_region(tissue_mask)

    print(x_min, y_min, x_max, y_max)

    stride = cf.stride_for_heatmap
    stride_rescale = int(stride / downsamples)

    set_of_pos = [(x , y) for x in range(x_min, x_max, stride_rescale) for y in range(y_min, y_max, stride_rescale)]

    set_of_real_pos = get_pos_of_patch_for_eval(tissue_mask, set_of_pos)

    set_of_real_pos = np.array(set_of_real_pos)

    print(set_of_real_pos.shape)

    col, row = slide.level_dimensions[level]
    thumbnail = slide.get_thumbnail((col, row))
    thumbnail = np.array(thumbnail)

    cv2.imwrite("thumbnail.jpg", thumbnail)

    draw_patch_pos_on_thumbnail(set_of_real_pos, thumbnail, downsamples)
