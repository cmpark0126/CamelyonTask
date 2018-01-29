import os, sys
import numpy as np
import openslide
import cv2
from xml.etree.ElementTree import parse
from PIL import Image

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
def _get_interest_region(slide, level, o_knl=5, c_knl=9):
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
    cv2.imwrite(output_dir + "/Level" + str(level) + "_ROI_OpenBW_int.jpg", thresh)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, close_knl)
   
    print('Generating rectangular mask...')
    
    thresh, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

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
    
    print('Rectangular mask generated.')

    cv2.imwrite("Data/Slide/ROI" , roi)

    return xmin, ymin, xmax-xmin, ymax-ymin

"""
param : slide name (string)
        downsamples (int)

return : annotations (list of numpy)
"""
def _get_annotation_from_xml(slide_name, downsamples):
    xml_name = slide_namei + ".xml"
    annotation_list = []
    annotation = []
    num_annotation = 0
    tree = etree.parse(xml_name)
    root = tree.getroot()

    for Annotation in root.iter("Annotation"):
        for Coordinate in Annotation.iter("Coordinate"):
            x = round(float(Coordinate.attrib["X"])/downsamples)
            y = round(float(Coordinate.attrib["Y"])/downsamples)
            annotation_list.append(x, y)
        contourlist.append(np.asarray(annotation_list))

    return annotation


"""
param : tumor_slide (openslide)
        level(int)
        annotations (list of numpy)
        

return : numpy array of tumor mask
"""
def _create_tumor_mask(tumor_slide, level, annotation):
    maskslide = np.zeros(tumor_slide.level_dimensions[level][::-1])
    cv2.drawContours(maskslide, contourlist, -1, 255, -1)
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
    thumbnail = slide.get_thumbnail(slide.level_dimensions(level))
    return thumbnail


"""
param : thumbnail (numpy arry)
        annotation (list of numpy)

use : _create_thumbnail(slide, level)

return : thumbnail (numpy array)
"""
def _draw_tumor_pos_on_thumbnail(thumbnail, annotation):
    cv2.drawContours(thumbnail, contourlist, -1, (0, 255, 0), 2)
    return thumbnail


"""
brief : 

param : thumbnail (numpy array)
        patch_pos (tuple(x, y, width, height))
        downsamples (int)
        
return : thumbnail (numpy array)

"""
def _draw_patch_pos_on_thunmbnail(thumbnail, patch_pos, downsamples):
    x, y, w, h = patch_pos
    cv2.rectangle(thubmnail, (int(x/downsamples), int(y/downsamples)), (int(x/downsamples) + int(w/downsamples), int(y/downsamples) + int(h/downsamples)))
    return thumbnail


"""
param : slide file (openslide)
        mask file (numpy)
        interest_region (tuple(x, y, width, height))
       
return : dataset(tuple(set of patch, set of information of patch))

"""
def _create_dataset(slide, mask, interest_region):
    return (set_of_patch, inform_of_patch)


if __name__ == "__main__":
    

    list_of_slidename = _get_tumor_slidename(ROOT, BASENAME)

    list_of_slidename = ["b_0"]

    for fn in list_of_slidename:
        slide = openslide.OpenSlide(fn)
        
        region = _get_interest_region(slide)

        annotations = _get_annotation_from_xml(fn, level)
        mask = _create_tumor_mask(annotations, level)
        
        patches, informs = _create_tumor_mask(annotations, mask, region)
        
        thumbnail = _create_thumbnail(slide, level)
        _draw_tumor_pos_on_thumbnail(thumbnail, annotations)
        _draw_patch_pos_on_thumbnail(thumbnail, patches)

          








