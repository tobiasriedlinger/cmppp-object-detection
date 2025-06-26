#!/usr/bin/env python3
"""
data loader
"""

import os
import numpy as np
from PIL import Image
from collections import namedtuple

from global_defs import CONFIG

Label = namedtuple('Label',['name','Id','trainId','color'])

class Cityscapes():  

    def __init__(self, split='val'):
        """
        Dataset loader that processes all images from one specified root directory
        Also searches for images in every subdirectory in root directory
        """

        prefix = os.path.join(CONFIG.IMG_DIR, 'leftImg8bit', split)
        
        self.images = []    # where to load input images - absolute paths
        self.targets = []   # where to load ground truth if available - absolute paths
        self.gt_inst = []   # where to load ground truth (instances)
        self.name = []      # image name
        self.sem_seg = []   # where to load semantic segmentation predictions - absolute paths
        self.pp = []        # where to load point process predictions - absolute paths
        self.bb = []        # where to load bounding box predictions - absolute paths

        for city in sorted(os.listdir(prefix)):
            for img in sorted(os.listdir(os.path.join(prefix,city))):

                self.images.append(os.path.join(prefix, city, img)) 
                self.targets.append(os.path.join(prefix.replace('leftImg8bit','gtFine'), city, img.replace('leftImg8bit','gtFine_labelTrainIds'))) 
                self.gt_inst.append(os.path.join(prefix.replace('leftImg8bit','gtFine'), city, img.replace('leftImg8bit','gtFine_instanceIds')))  
                self.name.append(img.split('_left')[0])
                self.sem_seg.append(os.path.join(CONFIG.SEM_SEG_DIR, img.replace('.png','.npy')))  
                self.pp.append(os.path.join(CONFIG.PP_DIR, img.replace('.png','.npy')))  
                self.bb.append(os.path.join(CONFIG.BB_DIR, img.replace('.png','.json')))  

    def __getitem__(self, index):
        """Generate one sample of data"""
        image = np.asarray(Image.open(self.images[index]).convert('RGB'))
        target = np.asarray(Image.open(self.targets[index])) 
        return image, target, self.gt_inst[index], self.name[index], self.sem_seg[index], self.pp[index], self.bb[index]

    def __len__(self):
        """Denote the total number of samples"""
        return len(self.images)


cs_labels = [
    #       name                       Id   trainId    color
    Label(  'unlabeled'            ,    0 ,     255 ,  (255,255,255) ),
    Label(  'ego vehicle'          ,    1 ,     255 ,  (  0,  0,  0) ),
    Label(  'rectification border' ,    2 ,     255 ,  (  0,  0,  0) ),
    Label(  'out of roi'           ,    3 ,     255 ,  (  0,  0,  0) ),
    Label(  'static'               ,    4 ,     255 ,  (  0,  0,  0) ),
    Label(  'dynamic'              ,    5 ,     255 ,  (111, 74,  0) ),
    Label(  'ground'               ,    6 ,     255 ,  ( 81,  0, 81) ),
    Label(  'road'                 ,    7 ,       0 ,  (128, 64,128) ),
    Label(  'sidewalk'             ,    8 ,       1 ,  (244, 35,232) ),
    Label(  'parking'              ,    9 ,     255 ,  (250,170,160) ),
    Label(  'rail track'           ,   10 ,     255 ,  (230,150,140) ),
    Label(  'building'             ,   11 ,       2 ,  ( 70, 70, 70) ),
    Label(  'wall'                 ,   12 ,       3 ,  (102,102,156) ),
    Label(  'fence'                ,   13 ,       4 ,  (190,153,153) ),
    Label(  'guard rail'           ,   14 ,     255 ,  (180,165,180) ),
    Label(  'bridge'               ,   15 ,     255 ,  (150,100,100) ),
    Label(  'tunnel'               ,   16 ,     255 ,  (150,120, 90) ),
    Label(  'pole'                 ,   17 ,       5 ,  (153,153,153) ),
    Label(  'polegroup'            ,   18 ,     255 ,  (153,153,153) ),
    Label(  'traffic light'        ,   19 ,       6 ,  (250,170, 30) ),
    Label(  'traffic sign'         ,   20 ,       7 ,  (220,220,  0) ),
    Label(  'vegetation'           ,   21 ,       8 ,  (107,142, 35) ),
    Label(  'terrain'              ,   22 ,       9 ,  (152,251,152) ),
    Label(  'sky'                  ,   23 ,      10 ,  ( 70,130,180) ),
    Label(  'person'               ,   24 ,      11 ,  (220, 20, 60) ),
    Label(  'rider'                ,   25 ,      12 ,  (255,  0,  0) ),
    Label(  'car'                  ,   26 ,      13 ,  (  0,  0,142) ),
    Label(  'truck'                ,   27 ,      14 ,  (  0,  0, 70) ),
    Label(  'bus'                  ,   28 ,      15 ,  (  0, 60,100) ),
    Label(  'caravan'              ,   29 ,     255 ,  (  0,  0, 90) ),
    Label(  'trailer'              ,   30 ,     255 ,  (  0,  0,110) ),
    Label(  'train'                ,   31 ,      16 ,  (  0, 80,100) ),
    Label(  'motorcycle'           ,   32 ,      17 ,  (  0,  0,230) ),
    Label(  'bicycle'              ,   33 ,      18 ,  (119, 11, 32) ),
    Label(  'license plate'        ,   -1 ,      -1 ,  (  0,  0,142) ),
]