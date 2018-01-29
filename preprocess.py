import os, sys
import numpy as np
import openslide
import cv2
import time
import multiprocessing

from functools import partial
from itertools import product
from itertools import repeat

from multiprocessing import Pool
from multiprocessing import Process
from xml.etree.ElementTree import parse
from PIL import Image

LEVEL = 4
#             width, height
PATCH_SIZE = (304, 304)

#INPUT_FN = "Tumor_" + TUMOR_NUMBER + ".tif"
#XML_FN = "Tumor_" + TUMOR_NUMBER + ".xml"

O_KNL = 5
C_KNL = 9

RECT = True

INPUT_FN = "b_1.tif"
XML_FN = "b_1.xml"

NUM_OF_SLIDE = 1

OUTPUT_PATCH_DIR ="DATA/"
OUTPUT_MASK_DIR = "DATA/"

def create_path(dir_name): 
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

def get_annotations_cord(xml_fn):
    annotation_list = []
    
    tree = parse(xml_fn)
    annotations = tree.getroot().find("Annotations")

    for anno in annotations.findall("Annotation"):
        coordinates = anno.find("Coordinates")
        coordinate_list = []
        for coord in coordinates.findall("Coordinate"):
            #                                       col                    row                    
            coordinate_list.append([float(coord.get("X")), float(coord.get("Y"))])
        annotation_list.append(coordinate_list)

    return np.asarray(annotation_list)


def create_patch(tumor_slide, slide_name, level=LEVEL, patch_size=PATCH_SIZE, output_dir=OUTPUT_PATCH_DIR):
    # x,   y
    col, row = tumor_slide.level_dimensions[0]

    downsamples = int(tumor_slide.level_downsamples[level])

    col = int(col / (patch_size[0] * downsamples))
    row = int(row / (patch_size[1] * downsamples))

    total_num = col * row
    cur_num = 0

    # we need to fix data image loss on right edge
    for c in range(col):
        for r in range(row):
            x = c * patch_size[0] * downsamples
            y = r * patch_size[1] * downsamples
            patch = tumor_slide.read_region((x, y), level, patch_size)
            patch.save(output_dir + "/"".png")
            cur_num = cur_num + 1
            print("\r" + str(multiprocessing.current_process) + slide_name + " Patch Create Percentage : %d / %d" % (
            cur_num, total_num), end="")

    print('\n')

    return True


def create_mask(tumor_slide, level, annotation_list, output_dir):
    # x,   y
    col, row = tumor_slide.level_dimensions[level]

    mask = np.zeros((row, col), np.uint8)

    divisor = tumor_slide.level_downsamples[level]

    for anno in annotation_list:
        for coord in anno:
            coord[0] = int(coord[0] / divisor)
            coord[1] = int(coord[1] / divisor) 
        pts = np.array(anno, np.int32)
        pts = pts.reshape((-1, 1, 2))
        cv2.fillPoly(mask, [pts], (255))
    
    # we need to transpose numpy list
    cv2.imwrite(output_dir + "/Level" + str(level) + ".jpg", mask) 
    
    return mask

def make_maskonslide(tumor_slide, mask, level, output_dir):
    row, col = mask.shape

    tumor_np = np.array(tumor_slide.get_thumbnail((col, row)))

    total_pixel = col * row
    cur_pixel = 0

    # we need to think how to use parallelism system with numpy
    for x in range(col):
        for y in range(row):
            if mask[y, x] == 255:
                tumor_np[y, x] = np.asarray([0, 0, 0])
                # print(tumor_np[y, x])
            cur_pixel = cur_pixel + 1
            print("\rPercenrage : %0.2f%% " %(cur_pixel / total_pixel * 100), end="")
    
    
    cv2.imwrite(output_dir + "/Level" + str(level) + "_MaskWithSlide.jpg", tumor_np)
    
    print('\n')

    return True

def makeROI(tumor_slide, level, output_dir, o_knl = 1, c_knl = 1, rect = False):

    col, row = tumor_slide.level_dimensions[level]
    
    ori_img = np.array(tumor_slide.read_region((0, 0), level, (col, row)))
    
    # color scheme change RGBA->RGB->HSV
    img = cv2.cvtColor(ori_img, cv2.COLOR_RGBA2RGB)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

    # Out of the HSV channels, only the saturation values are kept. (gray,
    # white, black pixels have low saturation values while tissue pixels
    # have high saturation)
    img = img[:,:,1]

    #roi[roi <= 150] = 0
    
    # Saturation values -> BW
    ret, thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    #cv2.imwrite(output_dir + "/Level" + str(level) + "_ROI_RawBW_int.jpg", roi)

    # Creation of opening and closing kernels
    open_knl = np.ones((o_knl, o_knl), dtype = np.uint8)
    close_knl = np.ones((c_knl, c_knl), dtype = np.uint8)

    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, open_knl)
    cv2.imwrite(output_dir + "/Level" + str(level) + "_ROI_OpenBW_int.jpg", thresh)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, close_knl)
    # If rect, make ROI rectangular
    if rect:
        print('Generating rectangular mask...')
        roi = make_rectmask(thresh, ori_img)
        print('Rectangular mask generated.')

    cv2.imwrite(output_dir + "/Level" + str(level) + "_ROI.jpg", roi)

    return roi

def make_rectmask(thresh, ori_img):
    print("in makeRECT")
    thresh, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnt = contours[0]
    #print(contours)
    ori_img = cv2.drawContours(ori_img, contours, -1, (0,255,0), 5)
    

    xmax = 0
    ymax = 0
    xmin = sys.maxsize
    ymin = sys.maxsize

    print("in makeRECT")
    for i in contours:
        x,y,w,h = cv2.boundingRect(i)
        if(x > xmax):
            xmax = x
        elif(x < xmin):
            xmin = x

        if(y > ymax):
            ymax = y
        elif(y < ymin):
            ymin = y
    
    cv2.rectangle(ori_img, (xmin, ymin), (xmax, ymax), (0,0,255), 5)

    print("in makeRECT")
    return ori_img


# main
if __name__ == "__main__":

    start_time = time.time()
    # define variable
    tumor_slide = openslide.OpenSlide(INPUT_FN)
    annotation_list = get_annotations_cord(XML_FN)

    # print information of Slide
    print("File name is " + INPUT_FN + "\n")
    print("level" + str(LEVEL), "size is", tumor_slide.level_dimensions[LEVEL], "\n")
    
    # check dir
    print(">> Check existence of dir\n")
    create_path(OUTPUT_PATCH_DIR)
    create_path(OUTPUT_MASK_DIR)
    print("done\n")

    #getting ROI
    print(">> Identifying ROI\n")
    makeROI(tumor_slide, LEVEL, OUTPUT_MASK_DIR, O_KNL, C_KNL,  RECT)
    print("done\n")

    print(">> Create Patch Slide : level" + str(LEVEL) + "\n")
    #create_patch_modifi(tumor_slide, LEVEL, PATCH_SIZE, OUTPUT_PATCH_DIR)
    print("done\n")

    print(">> Create Mask Image : level" + str(LEVEL) + "\n")
    #mask = create_mask(tumor_slide, LEVEL, annotation_list, OUTPUT_MASK_DIR)
    print("done\n")

    print(">> Place the Mask on Slide : level" + str(LEVEL) + "\n") 
    #make_maskonslide(tumor_slide, mask, LEVEL, OUTPUT_MASK_DIR)
    print("done\n")

    tumor_slide.close()
    print("--- %s seconds ---" %(time.time() - start_time))

