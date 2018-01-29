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
def _get_interest_region(slide):
    return x, y, width, height


"""
param : slide name (string)
        level (int)

return : annotations (list of numpy)
"""
def _get_annotation_from_xml(slide_name, level):
    return annotations


"""
param : annotations (list of numpy)

return : numpy array of tumor mask
"""
def _create_tumor_mask(annotations, level):
    return mask


"""
brief : if patch represent tumor label value is 1, else 0

param : mask (numpy)
        patch_pos (tuple(x, y, width, height))

return : label (int)

"""
def _determine_tumor(mask, patch_pos):
    return label


"""
param : slide file (openslide)
        level (int)
        
return : thumbnail (numpy array)

"""
def _create_thumbnail(slide, level):
    return thumbnail


"""
param : thumbnail (numpy arry)
        annotation (list of numpy)

use : _create_thumbnail(slide, level)

return : thumbnail (numpy array)
"""
def _draw_tumor_pos_on_thumbnail(slide, level, annotation):
    return thumbnail


"""
brief : 

param : thumbnail (numpy array)
        patch_pos (tuple(x, y, width, height))
        level (int)
        
return : thumbnail (numpy array)

"""
def _draw_patch_pos_on_thunmbnail(thumbnail):
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

          








