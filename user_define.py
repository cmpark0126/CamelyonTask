""" Created by DeepBio Camelyon toy problem team3
"""


class Config:
    '''config'''
    path_of_slide = './Data/slide'
    path_of_annotation = './Data/annotation'
    path_of_task_1 = './Data/task/task_1'

    # result of data preprocessing
    # './Data/result/$SLIDE_NAME/patch(etc)'
    path_for_generated_image = './Data/result'
    base_folder_for_patch = 'patch'
    base_folder_for_etc = 'etc'

    path_of_train_dataset = './Data/dataset/train'
    path_of_val_dataset = './Data/dataset/val'
    path_of_test_dataset = './Data/dataset/test'

    '''select option'''
    level_for_preprocessing = 4

    save_tissue_mask_image = True
    save_tumor_mask_image = True
    save_patch_images = False

    save_thumbnail_image = True

    # for create dataset
    key_of_data = 'data'
    key_of_informs = 'informations'

    list_of_slide_for_train = ['b_1',
                               'b_3',
                               'b_4',
                               'b_6',
                               'b_7',
                               'b_8',
                               'b_9',
                               'b_11',
                               'b_12',
                               'b_14']
    list_of_slide_for_val = ['b_2',
                             'b_5',
                             'b_10',
                             'b_13',
                             'b_15']

    # list_of_slide_for_train     = ['b_0',
    #                                'b_2',]
    # list_of_slide_for_val       = ['b_4',
    #                                'b_9',]


class Hyperparams:
    '''Hyper parameters'''
    # for data preprocess
    patch_size = (304, 304)

    # for dataset
    number_of_patch_per_slide = 2500
    ratio_of_tumor_patch      = 0.75
    threshold_of_tumor_rate   = 0.4

    # for run model
    ## resume from checkpoint
    resume        = True

    ## for optimizer
    learning_rate = 0.001
    momentum      = 0.9
    weight_decay  = 9e-4

    ## for epoch
    ### for train step
    batch_size_for_train = 250
    threshold_for_train = 0.1

    ### for eval step
    batch_size_for_eval = 250
    threshold_for_eval = 0.1
