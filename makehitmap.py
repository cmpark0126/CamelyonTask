import csv
from user_define import Config as cf
from user_define import Hyperparams as hp

import os

import numpy as np
import openslide
import cv2

f = open(cf.path_for_generated_image + "/result.csv",
         'r', encoding='utf-8')

target_path = os.path.join(cf.path_of_task_1, 'b_15.tif')
slide = openslide.OpenSlide(target_path)

print("start")

a = np.zeros(shape=slide.level_dimensions[4][::-1])

rdr = csv.reader(f)
for line in rdr:
    #print(line)
    if line[2] == '1.0':
        print(line[0], line[1])
        x_pos = line[0].strip()
        x = round(int(x_pos)/16)
        y_pos = line[1].strip()
        y = round(int(y_pos)/16)
        a[y:y+19, x:x+19] = 255


cv2.imwrite('reult.png', a)
