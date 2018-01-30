import os, sys
import os.path
import random
import numpy as np
import openslide
import cv2
from xml.etree.ElementTree import parse
from PIL import Image
import torch.utils.data as data

ROOT = './Data'
BASE_FOLDER_FOR_ANNOTATION = 'annotation'
BASE_FOLDER_FOR_SLIDE = 'slide'

class CAMELYON(data.Dataset):
    """
    CAMELYON Dataset preprocessed by DEEPBIO

    Args:
        root (string)
        level (int)
        patch_size (int, int)
    """
    base_folder_for_annotation = 'annotation'
    base_folder_for_slide = 'slide'
    base_folder_for_result = 'result'
    base_folder_for_patch = 'patch'
    base_folder_for_etc = 'etc'

    def __init__(self, root, slide_fn, xml_fn, level, patch_size, num_of_patch, ratio):

        self.root = os.path.expanduser(root)
        self.level = level
        self.slide_fn = slide_fn
        self.xml_fn = xml_fn

        self.patch_size = patch_size

        self.slide_path = os.path.join(self.root, self.base_folder_for_slide)
        self.xml_path = os.path.join(self.root, self.base_folder_for_annotation)

        self.patch_path = os.path.join(self.root, self.base_folder_for_result, self.slide_fn[:-4], self.base_folder_for_patch)
        self._check_path_existence(self.patch_path)

        self.etc_path = os.path.join(self.root, self.base_folder_for_result, self.slide_fn[:-4], self.base_folder_for_etc)
        self._check_path_existence(self.etc_path)

        self.slide = openslide.OpenSlide(os.path.join(self.slide_path, self.slide_fn))
        self.downsamples = int(self.slide.level_downsamples[self.level])
        self.annotation = self._get_annotation_from_xml()

        self.tissue_mask = self._create_tissue_mask()
        self.tumor_mask = self._create_tumor_mask()

        self.num_of_patch = num_of_patch
        self.ratio = ratio

        self.set_of_patch, self.set_of_pos = self._create_dataset()

        self.thumbnail = self._create_thumbnail()
        cv2.imwrite(os.path.join(self.etc_path, "thumbnail.jpg"), self.thumbnail)

    def _check_path_existence(self, dir_name):
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

    """
    param :

    return : tissue_mask (numpy_array)
    """
    def _create_tissue_mask(self, o_knl=5, c_knl=9):
        col, row = self.slide.level_dimensions[self.level]

        ori_img = np.array(self.slide.read_region((0, 0), self.level, (col, row)))

        # color scheme change RGBA->RGB->HSV
        img = cv2.cvtColor(ori_img, cv2.COLOR_RGBA2RGB)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

        # Out of the HSV channels, only the saturation values are kept. (gray,
        # white, black pixels have low saturation values while tissue pixels
        # have high saturation)
        img = img[:,:,1]

        #roi[roi <= 150] = 0

        # Saturation values -> BW
        ret, tissue_mask = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        #cv2.imwrite(output_dir + "/Level" + str(level) + "_ROI_RawBW_int.jpg", roi)

        # Creation of opening and closing kernels
        open_knl = np.ones((o_knl, o_knl), dtype = np.uint8)
        close_knl = np.ones((c_knl, c_knl), dtype = np.uint8)

        tissue_mask = cv2.morphologyEx(tissue_mask, cv2.MORPH_OPEN, open_knl)
        # cv2.imwrite(output_dir + "/Level" + str(level) + "_ROI_OpenBW_int.jpg", thresh)
        tissue_mask = cv2.morphologyEx(tissue_mask, cv2.MORPH_CLOSE, close_knl)

        return tissue_mask


    """
    param :

    return : annotations (list of numpy)
    """
    def _get_annotation_from_xml(self):

        annotation = []
        num_annotation = 0
        tree = parse(os.path.join(self.xml_path, self.xml_fn))
        root = tree.getroot()

        for Annotation in root.iter("Annotation"):
            annotation_list = []
            for Coordinate in Annotation.iter("Coordinate"):
                x = round(float(Coordinate.attrib["X"])/self.downsamples)
                y = round(float(Coordinate.attrib["Y"])/self.downsamples)
                annotation_list.append((x, y))
            annotation.append(np.asarray(annotation_list))

        return annotation


    """
    param : tumor_slide (openslide)
            level(int)
            annotations (list of numpy)

    return : numpy array of tumor mask
    """
    def _create_tumor_mask(self):
        tumor_mask = np.zeros(self.slide.level_dimensions[self.level][::-1])
        cv2.drawContours(tumor_mask, self.annotation, -1, 255, -1)
        cv2.imwrite("mask.jpg", tumor_mask)
        return tumor_mask


    """
    brief : if patch represent tumor label value is 1, else 0
    ROOT = "./Data"
    BASE_ANNO = "annotation"
    BASE_SLIDE = "slide"
    LEVEL = 4
    param : mask (numpy)
            patch_pos (tuple(x, y, width, height))
            percent (what percent will you determine as tumor)
            downsamples(int)
    return : label (int)

    """
    # def _determine_tumor(mask, patch_pos, percent, downsamples):
    #     x, y, w, h = patch_pos
    #     if percent > 1 or percent < 0:
    #         raise RuntimeError('Percent must be in 0 to 1')
    #     maskofpatch = mask[int(y/downsamples): int(y/downsamples) + int(h/downsamples), int(x/downsamples): int(x/downsamples) + int(w/downsamples)]
    #     if np.sum(maskofpatch) > percent * 255 * int(h/downsamples) * int(w/downsamples):
    #         label_num = 1
    #     else:
    #         label_num = 0
    #     return label_numROOT = "./Data"

    """
    param : slide file (openslide)
            num_of_patch (int)
            mask file (numpy)
            level (integer)
            patch_size (integer 2 tuple)

    return : set of position(list)
    """
    def _get_random_samples(self, mask):
        set_of_pos = []
        numberofregion = int(np.sum(mask)/255)

        if numberofregion < self.num_of_patch:
            raise RuntimeError('Random size is bigger than number of pixels in region')

        mask = np.reshape(mask, -1)
        sorting = np.argsort(mask)[::-1][:numberofregion]
        np.random.shuffle(sorting)
        dataset_number = sorting[:self.num_of_patch].astype(int)

        x, y = self.slide.level_dimensions[self.level]

        goleft = int(self.patch_size[0]/2)
        goup = int(self.patch_size[1]/2)

        for data in dataset_number:
            i = data % x - goleft
            j = data // x - goup
            set_of_pos.append((i * self.downsamples, j * self.downsamples, self.patch_size[0], self.patch_size[1]))

        print(len(set_of_pos))
        return set_of_pos

    """
    param : slide file (openslide)
            mask file (numpy)
            interest_region (tuple(x, y, width, height))
            num_of_patch (int)

    return : dataset(tuple(set of patch, set of pos of patch))

    """
    def _create_dataset(self, save_image=True):

        set_of_patch = []
        set_of_pos = []

        set_of_pos_intumor = self._get_random_samples(self.tumor_mask)
        set_of_pos_intissue = self._get_random_samples(self.tissue_mask)

        set_of_pos = set_of_pos_intumor + set_of_pos_intissue

        if save_image:
            i = 0
            for pos in set_of_pos:
                x, y, w, h = pos
                patch = self.slide.read_region((x, y), 0, (w, h))
                patch_fn = str(x)+"_"+str(y)+".png"
                patch.save(os.path.join(self.patch_path, patch_fn))
                i = i + 1
                print("\rPercentage : %d / %d" %(i, self.num_of_patch * 2), end="")
            print("\n")

        return set_of_patch, set_of_pos

    """
    param : slide file (openslide)
            level (int)

    return : thumbnail (numpy array)

    """
    def _create_thumbnail(self):
        col, row = self.slide.level_dimensions[self.level]

        thumbnail = self.slide.get_thumbnail((col, row))

        thumbnail = np.array(thumbnail)

        return thumbnail


    """
    param :

    use : _create_thumbnail(slide, level)

    return : thumbnail (numpy array)
    """
    def _draw_tumor_pos_on_thumbnail(self):
        cv2.drawContours(self.thumbnail, self.annotation, -1, (0, 255, 0), 4)
        cv2.imwrite(os.path.join(self.etc_path, "tumor_to_thumbnail.jpg"), self.thumbnail)

        self.thumbnail = self._create_thumbnail()

    """
    brief :

    param :

    return : thumbnail (numpy array)

    """
    def _draw_patch_pos_on_thumbnail(self):
        for pos in self.set_of_pos :
            x, y, w, h = pos
            cv2.rectangle(self.thumbnail, (int(x/self.downsamples), int(y/self.downsamples)), (int(x/self.downsamples) + int(w/self.downsamples), int(y/self.downsamples) + int(h/self.downsamples)),(0,0,255), 4)

        cv2.imwrite(os.path.join(self.etc_path, "patch_pos_to_thumbnail.jpg"), self.thumbnail)

        self.thumbnail = self._create_thumbnail()


"""
param : root (string)
        base_folder (string)

return : slide_list (string)
"""
def _get_file_list(usage):
    if usage == "slide":
        print("get list of file at" + os.path.join(ROOT, BASE_FOLDER_FOR_SLIDE))
        return os.listdir(os.path.join(ROOT, BASE_FOLDER_FOR_SLIDE))
    elif usage == "annotation":
        print("get list of file at" + os.path.join(ROOT, BASE_FOLDER_FOR_ANNOTATION))
        return os.listdir(os.path.join(ROOT, BASE_FOLDER_FOR_ANNOTATION))
    else:
        raise RuntimeError("invalid usage")


if __name__ == "__main__":

    list_of_slide = _get_file_list("slide")
    list_of_annotation = _get_file_list("annotation")

    print(list_of_slide)
    print(list_of_annotation)

    test = CAMELYON(ROOT, "b_2.tif", "b_2.xml", 4, (304, 304), 100, 0.8)
    test._draw_tumor_pos_on_thumbnail()
    test._draw_patch_pos_on_thumbnail()
