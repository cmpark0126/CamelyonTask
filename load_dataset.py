from __future__ import print_function
from PIL import Image
import os
import os.path
import errno
import numpy as np
import sys
import pickle

import torch.utils.data as data
#from .utils import download_url, check_integrity

class CAMELYON_DATALOADER(data.Dataset):
    """
    Args:
        root (string): Root directory of dataset where directory
        train (bool, optional): If True, creates dataset from training set, otherwise
            creates from test set.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
    """
    base_folder = 'dataset'

    def __init__(self, root, epoch, usage='train',
                 transform=None, target_transform=None,
                 download=False):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform
        self.usage = usage # training set or val set or test set

        self.dataset_list = self._get_dataset_list()

        # now load the picked numpy arrays
        if self.usage == 'train':
            filename = self.dataset_list[epoch % 2]
            fliepath = os.path.join(self.root, self.base_folder, self.usage, filename)
            fo = open(fliepath, 'rb')
            dataset = pickle.load(fo)
            fo.close()

            self.train_data = dataset['patch']
            self.train_labels = dataset['labels']

        elif self.usage == 'val':
            filename = self.dataset_list[epoch % 2]
            fliepath = os.path.join(self.root, self.base_folder, self.usage, filename)
            fo = open(fliepath, 'rb')
            dataset = pickle.load(fo)
            fo.close()

            self.val_data = dataset['patch']
            self.val_labels = dataset['labels']
        else:
            self.test_data = 0
            self.test_labels = 0

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        if self.usage == 'train':
            img, target = self.train_data[index], self.train_labels[index][0]
        elif self.usage == 'val':
            img, target = self.val_data[index], self.val_labels[index][0]
        else:
            img, target = self.test_data[index], self.test_labels[index][0]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        if self.usage == 'train':
            return len(self.train_data)
        elif self.usage == 'val':
            return len(self.val_data)
        else:
            return len(self.test_data)

    def _get_dataset_list(self):
        root = self.root
        usage = self.usage
        if usage == "train" or usage == "test" or usage == "val":
            dir_path = os.path.join("./Data", self.base_folder, usage)
            # print("get list of dataset binary file")
            file_list = os.listdir(dir_path)
            file_list.sort()
            return file_list
        else:
            raise RuntimeError("invalid usage")


def get_dataset(train_transform, test_transform, epoch=0):
    train_dataset = CAMELYON_DATALOADER('./Data',
                                        epoch,
                                        usage='train',
                                        download=False,
                                        transform=train_transform)
    val_dataset = CAMELYON_DATALOADER('./Data',
                                        epoch,
                                        usage='val',
                                        download=False,
                                        transform=test_transform)
    # test_dataset = CAMELYON_DATALOADER('./data',
    #                                     epoch,
    #                                     usage='test',
    #                                     download=False,
    #                                     transform=test_transform)
    test_dataset = 0

    return train_dataset, val_dataset, test_dataset

if __name__ == "__main__":
    get_dataset(None, None, 0)
