
import numpy as np
import time
import cv2
import openslide

from itertools import repeat

import multiprocessing

from multiprocessing import Pool
from multiprocessing import process

from preprocess import create_patch
from preprocess import create_path
from preprocess import create_mask

INPUT_PATH = "/mnt/disk2/interns/slide"

SLIDE_NAME =[  
                "b_1.tif",  "b_10.tif",  "b_11.tif",
                "b_12.tif", "b_13.tif", " b_14.tif",
                "b_15.tif", "b_2.tif",  "b_3.tif", 
                "b_4.tif", " b_5.tif",  "b_6.tif",
                "b_7.tif", " b_8.tif",  "b_9.tifi"
            ]


TEST_SLIDE_NAME = [ "b_1.tif", "b_2.tif"]

OUTPUT_PATH = "Data/Slide"

LEVEL = 0
PATCH_SIZE = (304, 304)

def open_slide(name):
    tumor_slide = openslide.OpenSlide(name)
    create_path(OUTPUT_PATH+str(name[:-4])+"/Patch")
    create_ROI(tumor_slide, LEVEL, OUTPUT_PATH+str(name[:-4])+"/Patch")
    #create_patch(tumor_slide, name, LEVEL, PATCH_SIZE, OUTPUT_PATH + str(name[0:-4]) + "/PATCH") 
    #return tumor_slide


def multiproc(nproc):
    pool = Pool(nproc)
    pool.map(open_slide, TEST_SLIDE_NAME)

# main
if __name__ == "__main__":
    start_time = time.time()
    
    for i in range(15):
        SLIDE_NAME[i] = INPUT_PATH + SLIDE_NAME[i] 
    #multiproc(15)
    print(SLIDE_NAME)

#multiproc for open_slide func
      
    procs = []
    for n in TEST_SLIDE_NAME:
        proc = multiprocessing.Process(target=open_slide, args=(n,))
        procs.append(proc)
        proc.start()

    for proc in procs:
        proc.join()
    

print("--- %s seconds ---" %(time.time() - start_time))

