import os
import numpy as np
import torch
import shutil
from tqdm import trange
from copy import deepcopy
from torchvision import datasets, transforms
from PIL import Image

from pycocotools import mask as coco_mask
from pycocotools.coco import COCO

import utils
from .baseset import base_set

COCO_PATH = os.path.join(utils.get_dataset_root(), "COCO2017")
# 2017 train images normalization constants
#   mean: 0.4700, 0.4468, 0.4076
#   sd: 0.2439, 0.2390, 0.2420

class COCOSeg(datasets.vision.VisionDataset):
    def __init__(self, root, train=True):
        super(COCOSeg, self).__init__(root, None, None, None)
        self.min_area = 200 # small areas are marked as crowded
        split_name = "train" if train else "val"
        self.annotation_path = os.path.join(root, 'annotations', 'instances_{}2017.json'.format(split_name))
        self.img_dir = os.path.join(root, '{}2017'.format(split_name))
        self.coco = COCO(self.annotation_path)
        self.img_ids = list(self.coco.imgs.keys())

        # COCO class
        class_list = sorted([i for i in self.coco.cats.keys()])

        # The instances labels in COCO dataset is not dense
        # e.g., total 80 classes. Some objects are labeled as 82
        # but they are 73rd class; while none is labeled as 83.
        self.class_map = {}
        for i in range(len(class_list)):
            self.class_map[class_list[i]] = i + 1
        
        # Given a class idx (1-80), self.instance_class_map gives the list of images that contain
        # this class idx
        class_map_dir = os.path.join(root, 'instance_seg_class_map', split_name)
        if not os.path.exists(class_map_dir):
            # Merge VOC and SBD datasets and create auxiliary files
            try:
                self.create_coco_class_map(class_map_dir)
            except (Exception, KeyboardInterrupt) as e:
                # Dataset creation fail for some reason...
                shutil.rmtree(class_map_dir)
                raise e
        
        self.instance_class_map = {}
        for c in range(1, 81):
            class_map_path = os.path.join(class_map_dir, str(c) + ".txt")
            with open(class_map_path, 'r') as f:
                class_idx_list = f.readlines()
            class_idx_list = [int(i.strip()) for i in class_idx_list if i]
            self.instance_class_map[c] = class_idx_list

        self.CLASS_NAMES_LIST = ['background']
        for i in range(len(class_list)):
            cls_name = self.coco.cats[class_list[i]]['name']
            self.CLASS_NAMES_LIST.append(cls_name)
    
    def create_coco_class_map(self, class_map_dir):
        assert not os.path.exists(class_map_dir)
        os.makedirs(class_map_dir)

        instance_class_map = {}
        for c in range(1, 81):
            instance_class_map[c] = []

        print("Computing COCO class-object masks...")
        for i in trange(len(self)):
            img_id = self.img_ids[i]
            mask = self._get_mask(img_id)
            contained_labels = torch.unique(mask)
            for c in contained_labels:
                c = int(c)
                if c == 0 or c == -1:
                    continue # background or ignore_mask
                instance_class_map[c].append(str(i)) # use string to format integer to write to txt
        
        for c in range(1, 81):
            with open(os.path.join(class_map_dir, str(c) + '.txt'), 'w') as f:
                f.write('\n'.join(instance_class_map[c]))

    def _get_img(self, img_id):
        img_desc = self.coco.imgs[img_id]
        img_fname = img_desc['file_name']
        img_fpath = os.path.join(self.img_dir, img_fname)
        return Image.open(img_fpath).convert('RGB')
    
    def _get_mask(self, img_id):
        img_desc = self.coco.imgs[img_id]
        img_fname = img_desc['file_name']
        label_fname = img_fname.replace('.jpg', '.png')
        img_fpath = os.path.join(self.img_dir, label_fname)
        return self._get_seg_mask(img_fpath)
    
    def __getitem__(self, idx: int):
        img_id = self.img_ids[idx]
        img = self._get_img(img_id)
        seg_mask = self._get_mask(img_id) # tensor
        return (img, seg_mask)

    def _get_seg_mask(self, fname: str):
        deleted_idx = [91, 83, 71, 69, 68, 66, 45, 30, 29, 26, 12]
        raw_lbl = np.array(Image.open(fname)).astype(np.int)
        ignore_idx = (raw_lbl == 255)
        raw_lbl += 1
        raw_lbl[raw_lbl > 91] = 0 # STUFF classes are mapped to background
        for d_idx in deleted_idx:
            raw_lbl[raw_lbl > d_idx] -= 1
        raw_lbl[ignore_idx] = -1
        return torch.tensor(raw_lbl)
    
    def get_class_map(self, class_id):
        return deepcopy((self.instance_class_map[class_id]))
    
    def get_label_range(self):
        return [i + 1 for i in range(80)]

    def __len__(self):
        return len(self.coco.imgs)

def get_train_set(cfg):
    ds = COCOSeg(COCO_PATH, True)
    return base_set(ds, "train", cfg)

def get_val_set(cfg):
    ds = COCOSeg(COCO_PATH, False)
    return base_set(ds, "test", cfg)