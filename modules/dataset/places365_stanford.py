import os
import json
from PIL import Image
import numpy as np
import torch
import torchvision
from torchvision import datasets, transforms

import utils
from .baseset import base_set

class Places365StanfordReader(datasets.vision.VisionDataset):
    '''
    Places365 scene classification task.

    Official Places365 project website: http://places2.csail.mit.edu/index.html

    This dataset reader implementation works on the "Small images (256 * 256) with easy directory structure" partition,
    which can be downloaded from http://data.csail.mit.edu/places/places365/places365standard_easyformat.tar
    '''

    def __init__(self, root, train=True):
        '''
        Initialize and load the ADE20K annotation file into memory.
        '''
        super(Places365StanfordReader, self).__init__(root, None, None, None)
        self.train = train

        self.base_dir = root
        self.base_dir = os.path.join(self.base_dir, 'places365standard_easyformat', 'places365_standard')

        if train:
            split = 'train'
        else:
            split = 'val'

        # Get class
        # we follow https://github.com/zhoubolei/places_devkit/blob/master/categories_places365.txt
        # s.t. class idx are assigned in an ascending manner
        self.CLASS_NAMES_LIST = sorted(os.listdir(os.path.join(self.base_dir, split)))

        # assign label to an image based on its path
        self.name_idx_dict = {}
        for i in range(len(self.CLASS_NAMES_LIST)):
            self.name_idx_dict[self.CLASS_NAMES_LIST[i]] = i

        with open(os.path.join(self.base_dir, '{}.txt'.format(split))) as f:
            self.img_path_list = f.readlines()

        self.img_path_list = [i.strip() for i in self.img_path_list]
    
    def path_to_label(self, path):
        """Get integer label of a sample from its path

        Args:
            path (str): complete or incomplete path of the image

        Returns:
            int: label consistent with https://github.com/zhoubolei/places_devkit/blob/master/categories_places365.txt
        """
        class_name = path.split('/')[-2]
        return self.name_idx_dict[class_name]
    
    def __getitem__(self, idx: int):
        """
        Args:
            key (int): key

        Returns:
            ret_dict
        """
        assert idx >= 0 and idx < len(self.img_path_list)
        img_path = os.path.join(self.base_dir, self.img_path_list[idx])
        raw_img = Image.open(img_path).convert('RGB')
        label = self.path_to_label(img_path)
        return (raw_img, label)

    def __len__(self):
        return len(self.img_path_list)

def get_train_set(cfg):
    ds = Places365StanfordReader(utils.get_dataset_root(), True)
    return base_set(ds, "train", cfg)

def get_val_set(cfg):
    ds = Places365StanfordReader(utils.get_dataset_root(), False)
    return base_set(ds, "test", cfg)
