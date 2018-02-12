import numpy as np
import openslide
import cv2
import torch
from PIL import Image
import os

from user_define import Config as cf
from user_define import Hyperparams as hp

if __name__ == "__main__":
    slide_fn = "t_5"

    thumnail_path = os.path.join(cf.path_for_result, slide_fn, slide_fn+"_thumbnail.jpg")
    thumbnail = Image.open(thumnail_path)

    predict_mask_path = os.path.join(cf.path_for_result, slide_fn, slide_fn + "_result.png")

    predict_mask = np.asarray(Image.open(predict_mask_path).convert('L'))
    biased = np.ones(predict_mask.shape)*255
    predict_mask = Image.fromarray(biased - predict_mask).convert('RGB')
    np_predict = np.array(predict_mask)

    np_predict[:, :,1] = 255

    predict_mask = Image.fromarray(np_predict)

    Image.blend(thumbnail, predict_mask, 0.5).save(os.path.join(cf.path_for_result, slide_fn, slide_fn + "_visual.png"))
