"""
Module containing reader to parse pascal_5i dataset from SBD and VOC2012
"""
import os
from copy import deepcopy
import torch
import torchvision

import utils
from .baseset import base_set
from .coco import COCOSeg

COCO_PATH = os.path.join(utils.get_dataset_root(), "COCO2017")

novel_dict = {
    0: [1, 5, 9, 13, 17, 21, 25, 29, 33, 37, 41, 45, 49, 53, 57, 61, 65, 69, 73, 77],
    1: [2, 6, 10, 14, 18, 22, 26, 30, 34, 38, 42, 46, 50, 54, 58, 62, 66, 70, 74, 78],
    2: [3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43, 47, 51, 55, 59, 63, 67, 71, 75, 79],
    3: [4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64, 68, 72, 76, 80]
}

class COCO20iReader(torchvision.datasets.vision.VisionDataset):
    def __init__(self, root, fold, split, exclude_novel=False, vanilla_label=False):
        """
        pascal_5i dataset reader

        Parameters:
            - root:  root to data folder containing SBD and VOC2012 dataset. See README.md for details
            - fold:  folding index as in OSLSM (https://arxiv.org/pdf/1709.03410.pdf)
            - split: Specify train/val split of VOC2012 dataset to read from. True indicates training
            - exclude_novel: boolean flag to indicate whether novel examples are removed or masked.
                There are two cases:
                    * If set to True, examples containing pixels of novel classes are excluded.
                        (Generalized FS seg uses this setting)
                    * If set to False, examples containing pixels of novel classes are included.
                        Novel pixels will be masked as background. (FS seg uses this setting)
                When train=False (i.e., evaluating), this flag is ignored: images containing novel
                examples are always selected.
        """
        super(COCO20iReader, self).__init__(root, None, None, None)
        assert fold >= 0 and fold <= 3
        assert split in [True, False]
        if vanilla_label:
            assert exclude_novel
        self.vanilla_label = vanilla_label

        # Get augmented VOC dataset
        self.vanilla_ds = COCOSeg(root, split)
        self.CLASS_NAMES_LIST = self.vanilla_ds.CLASS_NAMES_LIST

        # Split dataset based on folding. Refer to https://arxiv.org/pdf/1709.03410.pdf
        # Given fold number, define L_{test}
        self.val_label_set = novel_dict[fold]
        self.train_label_set = [i for i in range(
            1, 81) if i not in self.val_label_set]
        
        self.visible_labels = self.train_label_set
        self.invisible_labels = self.val_label_set
        
        # Pre-training or meta-training
        if exclude_novel:
            # Exclude images containing invisible classes and use rest
            novel_examples_list = []
            for label in self.invisible_labels:
                novel_examples_list += self.vanilla_ds.get_class_map(label)
            self.subset_idx = [i for i in range(len(self.vanilla_ds))]
            self.subset_idx = list(set(self.subset_idx) - set(novel_examples_list))
        else:
            # Use images containing at least one pixel from relevant classes
            examples_list = []
            for label in self.visible_labels:
                examples_list += self.vanilla_ds.get_class_map(label)
            self.subset_idx = list(set(examples_list))

        # Sort subset idx to make dataset deterministic (because set is unordered)
        self.subset_idx = sorted(self.subset_idx)

        # Generate self.class_map
        self.class_map = {}
        for c in range(1, 81):
            self.class_map[c] = []
            real_class_map = self.vanilla_ds.get_class_map(c)
            real_class_map_lut = {}
            for idx in real_class_map:
                real_class_map_lut[idx] = True
            for subset_i, real_idx in enumerate(self.subset_idx):
                if real_idx in real_class_map_lut:
                    self.class_map[c].append(subset_i)
        
        self.remap_dict = {}
        map_idx = 1
        for c in range(1, 81):
            if c in self.val_label_set:
                self.remap_dict[c] = 0 # novel classes are masked as background
            else:
                assert c in self.train_label_set
                self.remap_dict[c] = map_idx
                map_idx += 1
    
    def __len__(self):
        return len(self.subset_idx)
    
    def get_class_map(self, class_id):
        """
        class_id here is subsetted. (e.g., class_idx is 12 in vanilla dataset may get translated to 2)
        """
        assert class_id > 0
        assert class_id < (len(self.visible_labels) + 1)
        # To access visible_labels, we translate class_id back to 0-indexed
        return deepcopy(self.class_map[self.visible_labels[class_id - 1]])
    
    def get_label_range(self):
        return [i + 1 for i in range(len(self.visible_labels))]

    def __getitem__(self, idx: int):
        assert 0 <= idx and idx < len(self.subset_idx)
        img, target_tensor = self.vanilla_ds[self.subset_idx[idx]]
        if not self.vanilla_label:
            target_tensor = self.mask_pixel(target_tensor)
        return img, target_tensor

    def mask_pixel(self, target_tensor):
        """
        Following OSLSM, we mask pixels not in current label set as 0. e.g., when
        self.train = True, pixels whose labels are in L_{test} are masked as background

        Parameters:
            - target_tensor: segmentation mask (usually returned array from self.load_seg_mask)

        Return:
            - Offseted and masked segmentation mask
        """
        # Use the property that validation label split is contiguous to accelerate
        label_set = torch.unique(target_tensor)
        for l in label_set:
            l = int(l)
            if l == 0 or l == -1:
                continue # background and ignore_label are unchanged
            src_label = l
            target_label = self.remap_dict[l]
            target_tensor[target_tensor == src_label] = target_label
        return target_tensor

class PartialCOCOReader(torchvision.datasets.vision.VisionDataset):
    def __init__(self, root, split, exclusion_list):
        """
        Reader that partially reads the PASCAL dataset
        """
        super(PartialCOCOReader, self).__init__(root, None, None, None)
        self.vanilla_ds = COCOSeg(root, split)

        self.CLASS_NAMES_LIST = self.vanilla_ds.CLASS_NAMES_LIST

        self.label_list = []

        for l in self.vanilla_ds.get_label_range():
            if l in exclusion_list:
                continue
            self.label_list.append(l)
        
        self.label_list = sorted(self.label_list)

        self.subset_idx = [i for i in range(len(self.vanilla_ds))]
        self.subset_idx = set(self.subset_idx)

        for l in exclusion_list:
            self.subset_idx -= set(self.vanilla_ds.get_class_map(l))
        self.subset_idx = sorted(list(self.subset_idx))
    
    def __len__(self):
        return len(self.subset_idx)
    
    def get_class_map(self, class_id):
        """
        class_id here is subsetted. (e.g., class_idx is 12 in vanilla dataset may get translated to 2)
        """
        raise NotImplementedError
    
    def get_label_range(self):
        return deepcopy(self.label_list)

    def __getitem__(self, idx: int):
        assert 0 <= idx and idx < len(self.subset_idx)
        img, target_tensor = self.vanilla_ds[self.subset_idx[idx]]
        return img, target_tensor

def get_train_set(cfg):
    folding = cfg.DATASET.COCO20i.folding
    ds = COCO20iReader(COCO_PATH, folding, True, exclude_novel=True)
    return base_set(ds, "train", cfg)

def get_val_set(cfg):
    folding = cfg.DATASET.COCO20i.folding
    ds = COCO20iReader(COCO_PATH, folding, False, exclude_novel=False)
    return base_set(ds, "test", cfg)

def get_train_set_vanilla_label(cfg):
    folding = cfg.DATASET.COCO20i.folding
    ds = COCO20iReader(COCO_PATH, folding, True, exclude_novel=True, vanilla_label=True)
    return base_set(ds, "train", cfg)

def get_continual_train_set(cfg):
    ds = COCOSeg(COCO_PATH, True)
    return base_set(ds, "train", cfg)

def get_continual_test_set(cfg):
    ds = COCOSeg(COCO_PATH, False)
    return base_set(ds, "test", cfg)
