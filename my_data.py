import os, sys
import random
import numpy as np
import openslide
import cv2
from xml.etree.ElementTree import parse
from PIL import Image

ROOT = "./Data"
BASE_ANNO = "/annotation"
BASE_SLIDE = "/slide"
LEVEL = 4
"""
param : root (string)
        base_folder (string)

return : slide_list (string)
"""
def _get_tumor_slidename(root, base_folder):
    return slide_list


"""
param : slide file (openslide)
        level (int)

return : (x, y, width, height)
"""
def _create_tissue_mask(slide, level, o_knl=5, c_knl=9):
    col, row = slide.level_dimensions[level]

    ori_img = np.array(slide.read_region((0, 0), level, (col, row)))

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
    # cv2.imwrite(output_dir + "/Level" + str(level) + "_ROI_OpenBW_int.jpg", thresh)
    tissue_mask = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, close_knl)

    return tissue_mask

"""
param : slide name (string)
        downsamples (int)

return : annotations (list of numpy)
"""
def _get_annotation_from_xml(path_for_annotation, downsamples):
    # xml_name = slide_name + ".xml"
    annotation = []
    num_annotation = 0
    tree = parse(path_for_annotation)
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
param : tumor_slide (openslide)
        level(int)
        annotations (list of numpy)


return : numpy array of tumor mask
"""
def _create_tumor_mask(slide, level, annotation):
    maskslide = np.zeros(slide.level_dimensions[level][::-1])
    cv2.drawContours(maskslide, annotation, -1, 255, -1)
    cv2.imwrite("mask.jpg", maskslide)
    return maskslide


"""
brief : if patch represent tumor label value is 1, else 0

param : mask (numpy)
        patch_pos (tuple(x, y, width, height))
        percent (what percent will you determine as tumor)
        downsamples(int)
return : label (int)

"""
def _determine_tumor(mask, patch_pos, percent, downsamples):
    x, y, w, h = patch_pos
    if percent > 1 or percent < 0:
        raise RuntimeError('Percent must be in 0 to 1')
    maskofpatch = mask[int(y/downsamples): int(y/downsamples) + int(h/downsamples), int(x/downsamples): int(x/downsamples) + int(w/downsamples)]
    if np.sum(maskofpatch) > percent * 255 * int(h/downsamples) * int(w/downsamples):
        label_num = 1
    else:
        label_num = 0
    return label_num


"""
param : slide file (openslide)
        level (int)

return : thumbnail (numpy array)

"""
def _create_thumbnail(slide, level):
    col, row = slide.level_dimensions[level]

    thumbnail = slide.get_thumbnail((col, row))

    thumbnail = np.array(thumbnail)

    cv2.imwrite("thumbnail.jpg", thumbnail)

    return thumbnail


"""
param : thumbnail (numpy arry)
        annotation (list of numpy)

use : _create_thumbnail(slide, level)

return : thumbnail (numpy array)
"""
def _draw_tumor_pos_on_thumbnail(thumbnail, annotation):
    cv2.drawContours(thumbnail, annotation, -1, (0, 255, 0), 2)
    cv2.imwrite("tumor_to_thumbnail.jpg", thumbnail)
    return thumbnail


"""
brief :

param : thumbnail (numpy array)
        patch_pos (tuple(x, y, width, height))
        downsamples (int)

return : thumbnail (numpy array)

"""
def _draw_patch_pos_on_thumbnail(thumbnail, patch_pos, downsamples):
    length = len(patch_pos)
    for i in range(length):
        x, y, w, h = patch_pos[i]
        cv2.rectangle(thumbnail, (int(x/downsamples), int(y/downsamples)), (int(x/downsamples) + int(w/downsamples), int(y/downsamples) + int(h/downsamples)),(0,0,255), 1)
        print("\rPercentage %d / %d" %(i+1, length), end="")
    cv2.imwrite("patch_pos_to_thumbnail.jpg", thumbnail)
    print('\n')
    return thumbnail
"""
param : slide file (openslide)
        num_of_patch (int)
        mask file (numpy)
        level (integer)
        patch_size (integer 2 tuple)

return : set of position(list)
"""
def _get_random_samples(slide, num_of_patch, mask, level, patch_size):
    set_of_pos = []
    numberofregion = int(np.sum(mask)/255)
   
    if numberofregion < num_of_patch:
        raise RuntimeError('Random size is bigger than number of pixels in region')
   
    mask = np.reshape(mask, -1)
    sorting = np.argsort(mask)[::-1][:numberofregion]
    np.random.shuffle(sorting)
    dataset_number = sorting[:num_of_patch].astype(int)
    
    x = slide.level_dimensions[level][0]
    downsamples = int(slide.level_downsamples[level])
    goleft = int(patch_size[0]/2)
    goup = int(patch_size[1]/2)
    
    for data in dataset_number:
        i = data % x - goleft
        j = data // x - goup
        set_of_pos.append((i * downsamples, j * downsamples, patch_size[0], patch_size[1]))
        
    return set_of_pos

"""
param : slide file (openslide)
        mask file (numpy)
        interest_region (tuple(x, y, width, height))
        num_of_patch (int)
        ratio: ratio of tumor mask, 0~1 float

return : dataset(tuple(set of patch, set of pos of patch))

"""

def _create_dataset(slide, tumor_mask, tissue_mask, patch_size, num_of_patch, level, ratio): 

    set_of_patch = []
    set_of_pos = []
    patch_in_tumormask = int(num_of_patch * ratio)
    patch_in_tissuemask = num_of_patch - patch_in_tumormask


    set_of_pos_intumor = _get_random_samples(slide, num_of_patch, patch_in_tumormask, level, patch_size)
    set_of_pos_intissue = _get_random_samples(slide, num_of_patch, patch_in_tissuemask, level, patch_size)

    """
        for i in range(num_of_patch):
            x = random.randrange(pos_x, pos_x + width)
            y = random.randrange(pos_y, pos_y + height)
            patch = slide.read_region((x, y), 0, patch_size)
            patch.save("./PATCH/" + str(x)+"_"+str(y)+".png")
            # patch to numpy array
            set_of_pos.append((x, y, patch_size[0], patch_size[1]))
            print("\rPercentage : %d / %d" %(i+1, num_of_patch), end="")

        print("\n")
    """

    set_of_pos = set_of_pos_intumor + set_of_pos_intissue

    return set_of_patch, set_of_pos
'''
param:

retunrn: list of file name

'''
def _get_list():
    print("get list of file at" + ROOT + BASE_SLIDE)
    return os.listdir(ROOT + BASE_SLIDE)

if __name__ == "__main__":


    print(_get_list())
    # list_of_slidename = _get_tumor_slidename(ROOT, BASENAME)

    list_of_slidename = ["b_2"]
    for fn in list_of_slidename:
        root = os.path.expanduser(ROOT)
        print(root)
        path_for_slide = os.path.join(root, BASE_SLIDE, fn) + ".tif"
        path_for_annotation = os.path.join(root, BASE_ANNO, fn) + ".xml"

        print(path_for_slide)
        print(path_for_annotation)

        slide = openslide.OpenSlide("Data" + path_for_slide)
        downsamples = int(slide.level_downsamples[LEVEL])

        print(downsamples)

        tissue_mask = _create_tissue_mask(slide, LEVEL)

        print(region)

        annotation = _get_annotation_from_xml("Data" + path_for_annotation, downsamples)

        print(type(annotation))

        tumor_mask = _create_tumor_mask(slide, LEVEL, annotation)

        set_of_patch, set_of_pos = _create_dataset(slide, tumor_mask, tissue_mask, (304, 304), 1000, LEVEL)

        thumbnail = _create_thumbnail(slide, LEVEL)

        _draw_tumor_pos_on_thumbnail(thumbnail, annotation)
        _draw_patch_pos_on_thumbnail(thumbnail, set_of_pos, downsamples)
