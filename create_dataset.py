import os, sys
import random
import numpy as np
import openslide
from xml.etree.ElementTree import parse
import cv2

import pickle
import time

# for multiprocessing
from multiprocessing import Pool, Queue, Process, Array
from itertools import repeat

# user define variable
from user_define import Config as cf
from user_define import Hyperparams as hp


class CAMELYON_PREPRO():
    """
    CAMELYON Dataset preprocessed by DEEPBIO

    Args:
        slide_filename (string)
    """

    # config
    level = cf.level_for_preprocessing

    # hyper parameters
    patch_size              = hp.patch_size
    num_of_patch            = hp.number_of_patch_per_slide
    ratio_of_tumor_patch    = hp.ratio_of_tumor_patch
    threshold_of_tumor_rate = hp.threshold_of_tumor_rate

    def __init__(self, usage, slide_filename):
        target_slide_path   = os.path.join(cf.path_of_slide,
                                           slide_filename)
        self.slide          = openslide.OpenSlide(target_slide_path)
        self.downsamples    = int(self.slide.level_downsamples[self.level])

        xml_filename        = slide_filename[:-4]+".xml"
        target_xml_path     = os.path.join(cf.path_of_annotation,
                                           xml_filename)
        self.annotation     = self.get_annotation_from_xml(target_xml_path)

        # for save action
        self.patch_path     = os.path.join(cf.path_for_generated_image,
                                           slide_filename[:-4],
                                           cf.base_folder_for_patch)
        self.check_path(self.patch_path)

        self.etc_path       = os.path.join(cf.path_for_generated_image,
                                           slide_filename[:-4],
                                           cf.base_folder_for_etc)
        self.check_path(self.etc_path)

        # for create patch array
        self.tissue_mask        = self.create_tissue_mask(cf.save_tissue_mask_image)
        self.tumor_mask         = self.create_tumor_mask(cf.save_tumor_mask_image)

        num_of_patch_in_tumor_mask  = int(self.num_of_patch * self.ratio_of_tumor_patch)
        num_of_patch_in_tissue_mask = self.num_of_patch - num_of_patch_in_tumor_mask

        set_of_inform_in_tumor  = self.get_inform_of_random_samples(
                                    self.tumor_mask,
                                    num_of_patch_in_tumor_mask)

        set_of_inform_in_tissue = self.get_inform_of_random_samples(
                                    self.tissue_mask - self.tumor_mask,
                                    num_of_patch_in_tissue_mask)

        self.set_of_inform  = np.array(set_of_inform_in_tumor + set_of_inform_in_tissue)
        self.set_of_patch   = self.get_patch_data(cf.save_patch_images)

        self.create_dataset(usage, slide_filename)

        if cf.save_thumbnail_image:
            self.thumbnail = self.create_thumbnail()
            self.draw_tumor_pos_on_thumbnail()
            self.draw_patch_pos_on_thumbnail()

    """
    param :

    return : annotations (list of numpy)
    """
    def get_annotation_from_xml(self, target_xml_path):
        downsamples = self.downsamples

        annotation = []
        num_annotation = 0

        tree = parse(target_xml_path)
        root = tree.getroot()
        for Annotation in root.iter("Annotation"):
            annotation_list = []
            for Coordinate in Annotation.iter("Coordinate"):
                x = round(float(Coordinate.attrib["X"])/downsamples)
                y = round(float(Coordinate.attrib["Y"])/downsamples)
                annotation_list.append((x, y))
            annotation.append(np.asarray(annotation_list))

        return annotation

    """
    """
    def check_path(self, dir_name):
        path = ""

        while(True):
            split = dir_name.split('/', 1)

            path = path + split[0] + '/'

            if not os.path.isdir(path):
                os.mkdir(path, )
                print(path, "is created!")

            if len(split) == 1:
                break

            dir_name = split[1]

        return True

    """
    param :

    return : tissue_mask (numpy_array)
    """
    def create_tissue_mask(self, save_image=False):
        slide = self.slide
        level = self.level

        col, row = slide.level_dimensions[level]

        img = np.array(slide.read_region((0, 0), level, (col, row)))
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        img = img[:,:,1]

        _, tissue_mask = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        if save_image:
            target_image_path = os.path.join(self.etc_path,
                                             "tissue_mask.jpg")
            cv2.imwrite(target_image_path, tissue_mask)

        return tissue_mask

    """
    param :

    return : tumor mask (numpy_array)
    """
    def create_tumor_mask(self, save_image=False):
        slide       = self.slide
        level       = self.level
        annotation  = self.annotation

        col, row = slide.level_dimensions[level]
        tumor_mask = np.zeros((row, col))
        cv2.drawContours(tumor_mask, annotation, -1, 255, -1)

        if save_image:
            target_image_path = os.path.join(self.etc_path,
                                             "tumor_mask.jpg")
            cv2.imwrite(target_image_path, tumor_mask)

        return tumor_mask


    """
    param : mask (numpy)
            patch_pos (tuple(x, y, width, height))
            percent (what percent will you determine as tumor)
            downsamples(int)

    return : label (int)

    """
    def determine_tumor(self, patch_pos):
        downsamples = self.downsamples
        threshold   = self.threshold_of_tumor_rate
        tumor_mask  = self.tumor_mask

        min_x   = int(patch_pos[0] / downsamples)
        min_y   = int(patch_pos[1] / downsamples)

        width   = int(patch_pos[2] / downsamples)
        height  = int(patch_pos[3] / downsamples)

        max_x   = int(min_x + width)
        max_y   = int(min_y + height)

        area    = width * height

        if threshold > 1 or threshold < 0:
            raise RuntimeError('threshold must be in 0 to 1')

        #
        mask_of_patch = tumor_mask[min_y: max_y, min_x: max_x]

        #
        if np.sum(mask_of_patch) > (threshold * 255 * area):
            return 1
        else:
            return 0

    """
    param : slide file (openslide)
            num_of_patch (int)
            mask file (numpy)
            level (integer)
            patch_size (integer 2 tuple)

    return : set of position(list)
    """
    def get_inform_of_random_samples(self, mask, num_of_patch):
        slide               = self.slide
        level               = self.level
        downsamples         = self.downsamples
        patch_size          = self.patch_size

        set_of_inform       = []
        number_of_region    = int(np.sum(mask)/255)

        if number_of_region < num_of_patch:
            raise RuntimeError('Random size is bigger than number of pixels in region')

        mask = np.reshape(mask, -1)
        sorting = np.argsort(mask)[::-1][:number_of_region]
        np.random.shuffle(sorting)
        dataset_number = sorting[:num_of_patch].astype(int)

        width, _ = slide.level_dimensions[level]
        goleft = int(patch_size[0]/(2*downsamples))
        goup = int(patch_size[1]/(2*downsamples))

        for data in dataset_number:
            x = (data % width - goleft) * downsamples
            y = (data // width - goup) * downsamples

            is_tumor = self.determine_tumor((x, y, patch_size[0], patch_size[1]))
            set_of_inform.append([is_tumor, x, y, patch_size[0], patch_size[1]])

        return set_of_inform

    """
    param :

    return :

    """
    def get_patch_data(self, save_image=False):
        slide = self.slide
        num_of_patch = self.num_of_patch
        set_of_inform = self.set_of_inform
        set_of_patch = []

        i = 1

        if save_image:
            print("Save patch image")
            for pos in set_of_inform:
                is_tumor, x, y, w, h = pos
                patch = slide.read_region((x, y), 0, (w, h))
                set_of_patch.append(np.array(patch)[:, :, :3])

                # for image save
                patch_fn = str(x)+"_"+str(y)+"_"+str(is_tumor)+".png"
                target_image_path = os.path.join(self.patch_path,
                                                 patch_fn)
                patch.save(target_image_path)

                print("\rPercentage : %d / %d" %(i, num_of_patch), end="")
                i = i + 1

            print("\n")
        else:
            print("Do not save patch image")
            for pos in set_of_inform:
                is_tumor, x, y, w, h = pos
                patch = slide.read_region((x, y), 0, (w, h))
                set_of_patch.append(np.array(patch)[:, :, :3])

                print("\rPercentage : %d / %d" %(i, num_of_patch), end="")
                i = i + 1

            print("\n")

        set_of_patch = np.array(set_of_patch)

        return set_of_patch

    """
    param :

    return :
    """
    def create_dataset(self, usage, slide_filename):
        set_of_patch  = self.set_of_patch
        set_of_inform = self.set_of_inform

        dataset = {}

        dataset[cf.key_of_data]    = np.array(set_of_patch)
        dataset[cf.key_of_informs] = np.array(set_of_inform)

        if usage == 'train':
            fp = cf.path_of_train_dataset
        elif usage == 'val':
            fp = cf.path_of_val_dataset

        self.check_path(fp)

        fn = os.path.join(fp, slide_filename + ".pkl")
        fo = open(fn, 'wb')
        pickle.dump(dataset, fo, pickle.HIGHEST_PROTOCOL)
        fo.close()

        print("Done")

    """
    param : slide file (openslide)
            level (int)

    return : thumbnail (numpy array)

    """
    def create_thumbnail(self):
        col, row = self.slide.level_dimensions[self.level]
        thumbnail = self.slide.get_thumbnail((col, row))
        thumbnail = np.array(thumbnail)
        cv2.imwrite(os.path.join(self.etc_path, "thumbnail.jpg"), thumbnail)
        return thumbnail

    """
    param :

    use : create_thumbnail(slide, level)

    return : thumbnail (numpy array)
    """
    def draw_tumor_pos_on_thumbnail(self):
        cv2.drawContours(self.thumbnail, self.annotation, -1, (0, 255, 0), 4)
        cv2.imwrite(os.path.join(self.etc_path, "tumor_to_thumbnail.jpg"), self.thumbnail)

    """
    brief :

    param :

    return :

    """
    def draw_patch_pos_on_thumbnail(self):
        for pos in self.set_of_inform :
            is_tumor, x, y, w, h= pos
            if is_tumor:
                cv2.rectangle(self.thumbnail, (int(x/self.downsamples), int(y/self.downsamples)), (int(x/self.downsamples) + int(w/self.downsamples), int(y/self.downsamples) + int(h/self.downsamples)),(255,0,0), 4)
            else:
                cv2.rectangle(self.thumbnail, (int(x/self.downsamples), int(y/self.downsamples)), (int(x/self.downsamples) + int(w/self.downsamples), int(y/self.downsamples) + int(h/self.downsamples)),(0,0,255), 4)

        target_image_path = os.path.join(self.etc_path,
                                         "patch_pos_to_thumbnail.jpg")
        cv2.imwrite(target_image_path, self.thumbnail)

"""
param :

return :
"""
def get_file_list(usage):
    if usage == "slide" or usage == "annotation":
        print("get list of file at" + os.path.join("./Data", usage))
        file_list = os.listdir(os.path.join("./Data", usage))
        file_list.sort()
        return file_list
    else:
        raise RuntimeError("invalid usage")

"""
"""
def read_slide_and_save_bin(usage, list_of_slide, number):
    for slide in list_of_slide:
        print(slide, "on working...")
        data = CAMELYON_PREPRO(usage, slide)

"""
"""
def read_slide_and_save_bin_multi(usage, list_of_slide, number):

    #with Pool(processes = 10) as pool:
    #q = Queue()
    pool = Pool(8)
    print("pre")
    result = pool.starmap_async(CAMELYON_PREPRO, zip(repeat(usage), list_of_slide))

    print("after")

    result.wait()


"""
param : batch_size (int)

return :
"""
def create_train_and_val_dataset(number, num_of_slide_for_train):
    list_of_slide = get_file_list("slide")
    random.shuffle(list_of_slide)
    print(list_of_slide)

    if len(list_of_slide) < num_of_slide_for_train:
        raise RuntimeError("invalid param : num_of_slide_for_train must be smaller than number of file")

    list_of_slide_for_train = list_of_slide[:num_of_slide_for_train]
    list_of_slide_for_val = list_of_slide[num_of_slide_for_train:]

    print("create train dataset")
    read_slide_and_save_bin_multi("train",
                            list_of_slide_for_train,
                            number)

    print("create val dataset")
    read_slide_and_save_bin_multi("val",
                            list_of_slide_for_val,
                            number)

if __name__ == "__main__":
    start_time = time.time()
    create_train_and_val_dataset(0, 3)
    end_time = time.time()
    print( "Run time is :  ", end_time - start_time)
