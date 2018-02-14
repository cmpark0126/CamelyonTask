import csv
from user_define import Config as cf
from user_define import Hyperparams as hp

import os

import numpy as np
import openslide
import cv2

# user define variable
from user_define import Config as cf
from user_define import Hyperparams as hp

def create_heatmap(slide_fn):
    output_level = cf.level_for_preprocessing

    csv_path = os.path.join(cf.path_for_result, slide_fn, slide_fn + "_result.csv")
    print("input is ", csv_path)

    f = open(csv_path,
             'r', encoding='utf-8')

    target_path = os.path.join(cf.path_of_task_2, slide_fn + ".tif")
    slide = openslide.OpenSlide(target_path)

    print("start")

    output = np.zeros(shape=slide.level_dimensions[output_level][::-1])

    rdr = csv.reader(f)
    for line in rdr:
        if line[2] == '1.0':
            print(line[0], line[1])
            x_pos = line[0].strip()
            x = round(int(x_pos)/slide.level_downsamples[output_level])
            y_pos = line[1].strip()
            y = round(int(y_pos)/slide.level_downsamples[output_level])
            output[y:y+19, x:x+19] = 255

    target_path = os.path.join(cf.path_for_result, slide_fn, slide_fn + "_pred.png")
    print("out put is ", target_path)
    cv2.imwrite(target_path, output)


if __name__ == "__main__":

    for slide_fn in cf.list_of_slide_for_task2:
        create_heatmap(slide_fn)

    print("Done")
