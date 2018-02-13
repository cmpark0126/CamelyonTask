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

    def __init__(self, usage, slide_fn, pos, transform=None):

        #self.img = patch
        self.usage = usage
        self.slide = openslide.OpenSlide(slide_fn)
        self.pos = pos
        self.transform = transform

        if usage == 'train':
            self.path_of_dataset = cf.path_of_train_dataset
        elif usage == 'val':
            self.path_of_dataset = cf.path_of_val_dataset
        elif usage == 'test':
            self.path_of_dataset = cf.path_of_test_dataset
        else:
            raise RuntimeError("invalid usage")

        self.dataset_list = self._get_dataset_list(self.path_of_dataset)

        self.data = []
        self.labels = []

        if usage is 'train' or usage is 'val':
            print("train and val")
            for filename in self.dataset_list:
                fliepath = os.path.join(self.path_of_dataset, filename)
                fo = open(fliepath, 'rb')
                dataset = pickle.load(fo)

                self.data.append(dataset[cf.key_of_data])
                self.labels.append(dataset[cf.key_of_informs])

                fo.close()

            self.data = np.concatenate(self.data)
            print("data shape is ", self.data.shape)
            self.labels = np.concatenate(self.labels)
            print("label shape is ", self.labels.shape)


    def __getitem__(self, index):
        if self.usage is "test":
            target = self.pos[index]
            img = self.slide.read_region(target, 0, hp.patch_size).convert('RGB')

        elif self.usage is "train" or self.usage is "val" :
            img, target = self.data[index], self.labels[index][0]

        img = Image.fromarray(np.array(img))

        if self.transform is not None:
            img = self.transform(img)

        return img, target

    def __len__(self):
        if self.usage is 'test':
            return len(self.pos)
        else :
            return len(self.data)

    def _get_dataset_list(self, dir_path):
        file_list = os.listdir(dir_path)
        file_list.sort()
        return file_list


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

    set_of_real_pos = np.array(set_of_real_pos)

    return set_of_real_pos


def get_test_dataset(transform=None):
    start_time = time.time()
    set_of_real_pos = make_patch_imform()
    slide_fn = 't_4'
    target_path = os.path.join(cf.path_of_task_2, slide_fn + ".tif")
    test_dataset = CUSTOM_DATASET("test", target_path, set_of_real_pos, transform)
    end_time = time.time()
    print("creating dataset is end, Running time is :  ", end_time - start_time)
    return test_dataset

def get_train_dataset(transform=None):
    start_time = time.time()
    set_of_real_pos = make_patch_imform()
    slide_fn = 't_4'
    target_path = os.path.join(cf.path_of_task_2, slide_fn + ".tif")
    train_dataset = CUSTOM_DATASET("train",target_path, set_of_real_pos, transform)
    end_time = time.time()
    print("creating train dataset is end, Running time is :  ", end_time - start_time)
    return train_dataset

def get_val_dataset(transform=None):
    start_time = time.time()
    set_of_real_pos = make_patch_imform()
    slide_fn = 't_4'
    target_path = os.path.join(cf.path_of_task_2, slide_fn + ".tif")
    val_dataset = CUSTOM_DATASET("val", target_path, set_of_real_pos, transform)
    end_time = time.time()
    print("creating val dataset is end, Running time is :  ", end_time - start_time)
    return val_dataset


if __name__ == "__main__":
    start_time = time.time()
    test_dataset = get_train_dataset()

    end_time = time.time()
    print("creating dataset is end, Running time is :  ", end_time - start_time)
    print("Done")
