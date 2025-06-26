#!/usr/bin/env python3
'''
script including
class object with global settings
'''

class CONFIG:
  
    #---------------------#
    # set save path       #
    #---------------------#
  
    save_path   = '/work/user/output/' 
  
    #------------------#
    # select or define #
    #------------------#
  
    datasets    = ['cityscapes'] 
    model_names = ['DeepLab_rs50','HRnet'] 
    
    DATASET    = datasets[0]    
    MODEL_NAME = model_names[0]    

    #----------------------------#
    # paths for data preparation #
    #----------------------------#

    if DATASET == 'cityscapes':
        IMG_DIR     = '/work/cityscapes/'  
        if MODEL_NAME == 'DeepLab_rs50':
            SEM_SEG_DIR = '/work/semseg_preds/deeplab/'
            PP_DIR      = '/work/ppp_preds/deeplab/'
            BB_DIR      = '/work/cmpp_preds/deeplab/' 
        if MODEL_NAME == 'HRnet':
            SEM_SEG_DIR = '/work/semseg_preds/hrnet/'
            PP_DIR      = '/work/ppp_preds/hrnet/'
            BB_DIR      ='/work/cmpp_preds/hrnet/' 

    #--------------------------------------------------------------------#
    # select tasks to be executed by setting boolean variable True/False #
    #--------------------------------------------------------------------#

    VISUALIZE_PRED = False
    EVAL_CALIB     = False
    EVAL_PRED_BOX  = False

    #-----------#
    # optionals #
    #-----------#

    VIS_PRED_DIR = save_path + 'vis_pred/'      + DATASET + '/' + MODEL_NAME + '/'
    CALIB_DIR    = save_path + 'eval_calib/'    + DATASET + '/' + MODEL_NAME + '/'
    BOX_DIR      = save_path + 'eval_pred_box/' + DATASET + '/' + MODEL_NAME + '/'



