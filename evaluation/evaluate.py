#!/usr/bin/env python3
"""
main function
"""

import os

from global_defs  import CONFIG
from prepare_data import Cityscapes 
from functions    import vis_pred_i, compute_calib, detect_boxes


def main():

    """
    Load data
    """
    print('load dataset')

    if CONFIG.DATASET == 'cityscapes':
        loader = Cityscapes( )
    print('dataset:', CONFIG.DATASET)
    print('number of images: ', len(loader))


    """
    For visualizing the input data and predictions.
    """
    if CONFIG.VISUALIZE_PRED:
        print("visualize input data and predictions")

        if not os.path.exists( CONFIG.VIS_PRED_DIR ):
            os.makedirs( CONFIG.VIS_PRED_DIR )

        for i in range(len(loader)):
            vis_pred_i(loader[i])


    """
    For computing the calibration for semantic segmentation and point process.
    """
    if CONFIG.EVAL_CALIB:
        print("compute the calibration")

        box_size = 1000

        save_path = CONFIG.CALIB_DIR[:-1]+'_'+str(box_size)+'/'
        if not os.path.exists( save_path ):
            os.makedirs( save_path )

        compute_calib(loader, save_path, box_size)
    

    """
    For matching bounding boxes and point process.
    """
    if CONFIG.EVAL_PRED_BOX:
        print("detection of boxes")

        if not os.path.exists( CONFIG.BOX_DIR ):
            os.makedirs( CONFIG.BOX_DIR )

        box_size = 1000

        save_path = CONFIG.BOX_DIR[:-1]+'_'+str(box_size)+'/'
        if not os.path.exists( save_path ):
            os.makedirs( save_path )
        
        if not os.path.exists( os.path.join(save_path,'heatmaps') ):
            os.makedirs( os.path.join(save_path,'heatmaps') )
        
        detect_boxes(loader, save_path, box_size)


if __name__ == '__main__':
  
    print( "===== START =====" )
    main()
    print( "===== DONE! =====" )
    

