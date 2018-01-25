import os, sys
import numpy as np
import openslide
import cv2
from xml.etree.ElementTree import parse
from PIL import Image

TUMOR_NUMBER = "029"

LEVEL = 4
#             width, height
PATCH_SIZE = (240, 240)

INPUT_FN = "Tumor_" + TUMOR_NUMBER + ".tif"
XML_FN = "Tumor_" + TUMOR_NUMBER + ".xml"

OUTPUT_PATCH_DIR ="DATA/PATCH/Tumor_"+ TUMOR_NUMBER + "/Level" + str(LEVEL)
OUTPUT_MASK_DIR = "DATA/MASK/Tumor_" + TUMOR_NUMBER + "/Level" + str(LEVEL)

def PathCreator(dir_name): 
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

def GetAnnotationsCoordinate(xml_fn):
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


def CreatePatchSlide(tumor_slide, level, patch_size, output_dir):
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
            x = c * patch_size[0]
            y = r * patch_size[1]
            patch = tumor_slide.read_region((x * downsamples, y * downsamples), level, patch_size)
            patch.save(output_dir + "/" + str(x) + "_" + str(y) + ".png")
            cur_num = cur_num + 1
            print("\rPercentage : %d / %d" %(cur_num, total_num), end="")
    
    print('\n')

    return True


def CreateMaskSlide(tumor_slide, level, annotation_list, output_dir):
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

def PlaceMaskonSlide(tumor_slide, mask, level, output_dir):
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


# main
if __name__ == "__main__":

    # define variable
    tumor_slide = openslide.OpenSlide(INPUT_FN)

    annotation_list = GetAnnotationsCoordinate(XML_FN)

    # print information of Slide
    print("File name is " + INPUT_FN + "\n")
    print("level" + str(LEVEL), "size is", tumor_slide.level_dimensions[LEVEL], "\n")
    
    # check dir
    print(">> Check existence of dir\n")
    PathCreator(OUTPUT_PATCH_DIR)
    PathCreator(OUTPUT_MASK_DIR)
    print("done\n")
    
    print(">> Create Patch Slide : level" + str(LEVEL) + "\n")
    CreatePatchSlide(tumor_slide, LEVEL, PATCH_SIZE, OUTPUT_PATCH_DIR)
    print("done\n")

    print(">> Create Mask Image : level" + str(LEVEL) + "\n")
    mask = CreateMaskSlide(tumor_slide, LEVEL, annotation_list, OUTPUT_MASK_DIR)
    print("done\n")

    print(">> Place the Mask on Slide : level" + str(LEVEL) + "\n") 
    PlaceMaskonSlide(tumor_slide, mask, LEVEL, OUTPUT_MASK_DIR)
    print("done\n")

    tumor_slide.close()
