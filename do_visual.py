import numpy as np
import openslide
import cv2
import torch
from PIL import Image
import os

from user_define import Config as cf
from user_define import Hyperparams as hp

dim = 4

if __name__ == "__main__":
    slide_fn = "t_4"

    thumnail_path = os.path.join(cf.path_for_generated_image, slide_fn, slide_fn+"_thumbnail.jpg")
    thumbnail = Image.open(thumnail_path)

    predict_mask_path = os.path.join(cf.path_for_generated_image, slide_fn, slide_fn + "_result.png")

    predict_mask = np.asarray(Image.open(predict_mask_path).convert('L'))
    biased = np.ones(predict_mask.shape)*255
    predict_mask = Image.fromarray(biased - predict_mask).convert('RGB')
    np_predict = np.array(predict_mask)

    np_predict[:, :,1] = 255
    #np_predict[2] = 0



    predict_mask = Image.fromarray(np_predict)
    """
    print(predict_mask.sum())
    color_change = np.zeros(thumbnail.size[::-1])

    print(predict_mask.shape)
    print(color_change.shape)

    predict_mask = np.expand_dims(predict_mask, axis=2)
    color_change = np.expand_dims(color_change, axis=2)
    predict_mask = np.concatenate((color_change, predict_mask, color_change), axis = 2)
    print(predict_mask.shape)
    print(np.asarray(thumbnail).shape)
    """
    Image.blend(thumbnail, predict_mask, 0.2).save(slide_fn + '_visual.png')
