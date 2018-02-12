from __future__ import print_function
from PIL import Image
import os
import os.path
import errno
import numpy as np
import sys
import pickle
import openslide

import torch.utils.data as data

# user define variable
from user_define import Config as cf
from user_define import Hyperparams as hp

from remove_background import *


class CUSTOM_DATASET(data.Dataset):

    def __init__(self, patch, pos, transform=None):

        #self.img = patch
        self.slide = openslide.OpenSlide(patch)
        self.pos = pos
        self.transform = transform

    def __getitem__(self, index):
        #img, pos = self.img[index], self.pos[index]
        pos = self.pos[index]
        img = self.slide.read_region(pos, 0, hp.patch_size).convert('RGB')

        img = Image.fromarray(np.array(img))

        if self.transform is not None:
            img = self.transform(img)

        return img, pos

    def __len__(self):
        return len(self.pos)


def make_patch_imform():
    slide_fn = 't_4'
    target_path = os.path.join(cf.path_of_task_2, slide_fn + ".tif")
    slide = openslide.OpenSlide(target_path)

    level = cf.level_for_preprocessing
    downsamples = int(slide.level_downsamples[level])

    tissue_mask = create_tissue_mask(slide)
    x_min, y_min, x_max, y_max = get_interest_region(tissue_mask)

    stride = cf.stride_for_heatmap
    stride_rescale = int(stride / downsamples)

    set_of_pos = [(x, y) for x in range(x_min, x_max, stride_rescale)
                  for y in range(y_min, y_max, stride_rescale)]
    set_of_real_pos = get_pos_of_patch_for_eval(
        target_path, tissue_mask, set_of_pos)
    #set_of_patch, set_of_real_pos = get_pos_of_patch_for_eval(slide, tissue_mask, set_of_pos)
    # print(set_of_patch)
    # print(type(set_of_patch))

    set_of_real_pos = np.array(set_of_real_pos)
    #set_of_patch = np.array(set_of_patch)

    # return  set_of_patch ,set_of_real_pos
    return set_of_real_pos


def get_test_dataset(transform=None):

    start_time = time.time()
    #set_of_patch, set_of_real_pos = make_patch_imform()
    set_of_real_pos = make_patch_imform()
    slide_fn = 't_4'
    target_path = os.path.join(cf.path_of_task_2, slide_fn + ".tif")
    slide = openslide.OpenSlide(target_path)
    test_dataset = CUSTOM_DATASET(target_path, set_of_real_pos, transform)
    end_time = time.time()
    print("creating dataset is end, Running time is :  ", end_time - start_time)
    return test_dataset


if __name__ == "__main__":
    start_time = time.time()
    test_dataset = get_test_dataset()

    end_time = time.time()
    print("creating dataset is end, Running time is :  ", end_time - start_time)
    print("Done")
