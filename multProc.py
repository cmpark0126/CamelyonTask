
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

SLIDE_NAME =[  
                "b_1.tif",  "b_10.tif",  "b_11.tif",
                "b_12.tif", "b_13.tif", " b_14.tif",
                "b_15.tif", "b_2.tif",  "b_3.tif", 
                "b_4.tif", " b_5.tif",  "b_6.tif",
                "b_7.tif", " b_8.tif",  "b_9.tifi"
            ]

TEST_SLIDE_NAME = [ "b_1.tif", "b_2.tif"]

LEVEL = 0
PATCH_SIZE = (304, 304)

def open_slide(name):
    tumor_slide = openslide.OpenSlide(name)
    create_path("DATA/PATCH/Tumor_"+str(name)+"/Level"+str(LEVEL))
    create_patch(tumor_slide, name, LEVEL, PATCH_SIZE, "DATA/PATCH/Tumor_"+str(name)+"/Level"+str(LEVEL)) 
    #return tumor_slide


def multiproc(nproc):
    pool = Pool(nproc)
    pool.map(open_slide, TEST_SLIDE_NAME)
    '''
    tumor_slide = openslide.OpenSlide("b_1.tif")
    tumor_slide2 = openslide.OpenSlide("b_2.tif")

    create_path("DATA/PATCH/Tumor_"+"b_1.tif"+"/Level"+str(LEVEL))
    create_path("DATA/PATCH/Tumor_"+"b_2.tif"+"/Level"+str(LEVEL))
    
    r = [
            [tumor_slide, "b_1", PATCH_SIZE, "DATA/PATCH/Tumor_"+"b_1"+"/Level"+str(LEVEL)],
            [tumor_slide2, "b_2", PATCH_SIZE, "DATA/PATCH/Tumor_"+"b_2"+"/Level"+str(LEVEL)]
        ]
        
    #pool.starmap(create_patch, [r, repeat("test"), repeat(PATCH_SIZE), repeat("DATA/PATCH/Tumor_"+"test"+"/Level"+str(LEVEL))])
    pool.starmap(create_patch, r)
    '''

# main
if __name__ == "__main__":
    start_time = time.time()
#    multiproc(2)

#multiproc for open_slide func
    '''  
    procs = []
    for n in TEST_SLIDE_NAME:
        proc = multiprocessing.Process(target=open_slide, args=(n,))
        procs.append(proc)
        proc.start()

    for proc in procs:
        proc.join()
    '''
    tumor_slide = openslide.OpenSlide("b_1.tif")
    tumor_slide2 = openslide.OpenSlide("b_2.tif")
    
    tumor_list = [tumor_slide, tumor_slide2]
    create_path("DATA/PATCH/Tumor_"+"test"+"/Level"+str(LEVEL))
 

#multiproc for create_patch
    procs = []
    for n in tumor_list:
        proc = multiprocessing.Process(target=create_patch, args=(n,"name",LEVEL, PATCH_SIZE,"DATA/PATCH/Tumor_"+"test"+"/Level"+str(LEVEL)))
        procs.append(proc)
        proc.start()

    for proc in procs:
        proc.join()

    print("--- %s seconds ---" %(time.time() - start_time))


