from __future__ import print_function
from PIL import Image
import os
import os.path
import errno
import numpy as np
import sys
import pickle

import torch.utils.data as data

# user define variable
from user_define import Config as cf
from user_define import Hyperparams as hp

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

    def __init__(self, usage='train',
                 transform=None, target_transform=None):
        self.transform = transform
        self.target_transform = target_transform
        self.usage = usage

        if usage == 'train':
            self.path_of_dataset = cf.path_of_train_dataset
        elif usage == 'val':
            self.path_of_dataset = cf.path_of_val_dataset
        elif usage == 'test':
            self.path_of_dataset = cf.path_of_test_dataset
        else:
            raise RuntimeError("invalid usage")

        self.dataset_list = self._get_dataset_list(self.path_of_dataset)

        print(self.dataset_list)
        # now load the picked numpy arrays
        self.data = []
        self.labels = []
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
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        if self.usage == 'test':
            img, target = self.data[index], self.labels[index]
        else:
            img, target = self.data[index], self.labels[index][0]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.data)

    def _get_dataset_list(self, dir_path):
        file_list = os.listdir(dir_path)
        file_list.sort()
        return file_list


def get_train_dataset(train_transform, test_transform,
                      train_target_transform=None, test_target_transform=None):
    train_dataset = CAMELYON_DATALOADER(usage='train',
                                        transform=train_transform,
                                        target_transform=train_target_transform)
    return train_dataset

def get_val_dataset(train_transform, test_transform,
                      train_target_transform=None, test_target_transform=None):

    val_dataset = CAMELYON_DATALOADER(usage='val',
                                      transform=test_transform,
                                      target_transform=test_target_transform)
    return val_dataset

def get_test_dataset(train_transform, test_transform,
                      train_target_transform=None, test_target_transform=None):
    test_dataset = CAMELYON_DATALOADER(usage='test',
                                       transform=test_transform,
                                       target_transform=test_target_transform)
    return test_dataset

if __name__ == "__main__":
    get_train_dataset(None, None)
    get_val_dataset(None, None)
    get_test_dataset(None, None)
    print("Dane")
