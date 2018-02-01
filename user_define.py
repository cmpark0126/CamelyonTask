""" Created by DeepBio Camelyon toy problem team3
"""
class Config:
    '''config'''
    path_of_slide               = './Data/slide'
    path_of_annotation          = './Data/annotation'

    # result of data preprocessing
    # './Data/result/$SLIDE_NAME/patch(etc)'
    path_for_generated_image    = './Data/result'
    base_folder_for_patch       = 'patch'
    base_folder_for_etc         = 'etc'

    path_of_train_dataset       = './Data/dataset/train'
    path_of_val_dataset         = './Data/dataset/val'
    path_of_test_dataset        = './Data/dataset/test'

    '''select option'''
    level_for_preprocessing     = 4

    save_tissue_mask_image      = True
    save_tumor_mask_image       = True
    save_patch_images           = False

    save_thumbnail_image        = True

    # for create dataset
    key_of_data                 = 'data'
    key_of_informs              = 'informations'



class Hyperparams:
    '''Hyper parameters'''
    # for data preprocess
    patch_size                  = (304, 304)

    # for train dataset
    number_of_patch_per_slide   = 500
    ratio_of_tumor_patch        = 0.5
    threshold_of_tumor_rate     = 0.3

    # for test dataset
