import csv
from user_define import Config as cf
from user_define import Hyperparams as hp

import os

import numpy as np
import openslide
import cv2


slide_fn = 'b_1'
output_level = 4

f = open(cf.path_for_generated_image + "/" + slide_fn + "_result.csv",
         'r', encoding='utf-8')

target_path = os.path.join(cf.path_of_slide, slide_fn + ".tif")
slide = openslide.OpenSlide(target_path)

print("start")

output = np.zeros(shape=slide.level_dimensions[output_level][::-1])

rdr = csv.reader(f)
for line in rdr:
    print(line)
    
    if line[2] == '1.0':
        print(line[0], line[1])
        x_pos = line[0].strip()
        x = round(int(x_pos)/slide.level_downsamples[output_level])
        y_pos = line[1].strip()
        y = round(int(y_pos)/slide.level_downsamples[output_level])
        output[y:y+19, x:x+19] = 255
    

cv2.imwrite(slide_fn + '_reult.png', output)
