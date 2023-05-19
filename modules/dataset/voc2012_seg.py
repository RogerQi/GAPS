import os
import shutil
from PIL import Image
from scipy.io import loadmat
from copy import deepcopy
import numpy as np
import torch
from torchvision import datasets, transforms

import utils
from .baseset import base_set

def maybe_download_voc(root):
    '''
    Helper function to download the Pascal VOC and SBD dataset
    '''
    # Use torchvision download routine to download dataset to root
    sbd_path = os.path.join(root, 'sbd')
    from torchvision import datasets
    if not os.path.exists(sbd_path):
        sbd_set = datasets.SBDataset(sbd_path, image_set='train', mode="segmentation", download=True)
    if not os.path.exists(os.path.join(root, 'VOCdevkit')):
        voc_set = datasets.VOCSegmentation(root, image_set='train', download = True)

def load_seg_mask(file_path):
    """
    Load seg_mask from file_path (supports .mat and .png).

    Target masks in SBD are stored as matlab .mat; while those in VOC2012 are .png

    Parameters:
        - file_path: path to the segmenation file

    Return: a numpy array of dtype long and element range(0, 21) containing segmentation mask
    """
    if file_path.endswith('.mat'):
        mat = loadmat(file_path)
        target = Image.fromarray(mat['GTcls'][0]['Segmentation'][0])
    else:
        target = Image.open(file_path)
    target_np = np.array(target, dtype=np.int16)

    # in VOC, 255 is used as ignore_mask in many works
    target_np[target_np > 20] = -1
    return target_np

def create_pascal_voc_aug(root):
    # Define base to SBD and VOC2012
    sbd_base = os.path.join(root, 'sbd')
    voc_base = os.path.join(root, 'VOCdevkit', 'VOC2012')

    # Define path to relevant txt files
    sbd_train_list_path = os.path.join(root, 'sbd', 'train.txt')
    sbd_val_list_path   = os.path.join(root, 'sbd', 'val.txt')
    voc_train_list_path = os.path.join(
        voc_base, 'ImageSets', 'Segmentation', 'train.txt')
    voc_val_list_path   = os.path.join(
        voc_base, 'ImageSets', 'Segmentation', 'val.txt')

    # Use np.loadtxt to load all train/val sets
    sbd_train_set = set(np.loadtxt(sbd_train_list_path, dtype="str"))
    sbd_val_set   = set(np.loadtxt(sbd_val_list_path, dtype="str"))
    voc_train_set = set(np.loadtxt(voc_train_list_path, dtype="str"))
    voc_val_set   = set(np.loadtxt(voc_val_list_path, dtype="str"))
    
    trainaug_set = (sbd_train_set | sbd_val_set | voc_train_set) - voc_val_set
    val_set = voc_val_set
    
    new_base_dir = os.path.join(root, 'PASCAL_SBDAUG')
    assert not os.path.exists(new_base_dir)
    os.makedirs(new_base_dir)
    
    new_img_dir = os.path.join(new_base_dir, "raw_images")
    new_ann_dir = os.path.join(new_base_dir, "annotations")
    os.makedirs(new_img_dir)
    os.makedirs(new_ann_dir)
    
    def merge_and_save(name_list, class_map_dir, split_name):
        class_map = {}
        # There are 20 (foreground) classes in VOC Seg
        for c in range(1, 21):
            class_map[c] = []

        for name in name_list:
            if name in voc_train_set or name in voc_val_set:
                # Prefer VOC annotation than the SBD annotation
                image_path = os.path.join(voc_base, 'JPEGImages', name + '.jpg')
                ann_path = os.path.join(voc_base, 'SegmentationClass', name + '.png')
            else:
                # If VOC annotation is not available, use SBD annotation
                image_path = os.path.join(sbd_base, 'img', name + '.jpg')
                ann_path = os.path.join(sbd_base, 'cls', name + '.mat')

            new_img_path = os.path.join(new_img_dir, name + '.jpg')
            new_ann_path = os.path.join(new_ann_dir, name + '.npy')
            shutil.copyfile(image_path, new_img_path)
            seg_mask_np = load_seg_mask(ann_path)

            # Test if class c is in the image
            for c in range(1, 21):
                if c in seg_mask_np:
                    class_map[c].append(name)

            with open(new_ann_path, 'wb') as f:
                np.save(f, seg_mask_np)

        # Save auxiliary txt files
        # Save class map (for PASCAL-5i)
        class_map_dir = os.path.join(new_base_dir, class_map_dir)
        assert not os.path.exists(class_map_dir)
        os.makedirs(class_map_dir)
        for c in range(1, 21):
            with open(os.path.join(class_map_dir, str(c) + '.txt'), 'w') as f:
                f.write('\n'.join(class_map[c]))
        
        # Save set pointers
        with open(os.path.join(new_base_dir, split_name + '.txt'), 'w') as f:
            f.write('\n'.join(name_list))
    
    merge_and_save(trainaug_set, os.path.join(new_base_dir, "trainaug_class_map"), 'trainaug')
    merge_and_save(val_set, os.path.join(new_base_dir, "val_class_map"), 'val')

class PascalVOCSegReader(datasets.vision.VisionDataset):
    """
    pascal_5i dataset reader

    Parameters:
        - root:  root to data folder containing SBD and VOC2012 dataset. See README.md for details
        - fold:  folding index as in OSLSM (https://arxiv.org/pdf/1709.03410.pdf)
        - train: a bool flag to indicate whether L_{train} or L_{test} should be used
    """

    CLASS_NAMES_LIST = [
        "background",
        "aeroplane",
        "bicycle",
        "bird",
        "boat",
        "bottle",
        "bus",
        "car",
        "cat",
        "chair",
        "cow",
        "diningtable",
        "dog",
        "horse",
        "motorbike",
        "person",
        "potted plant",
        "sheep",
        "sofa",
        "train",
        "tv/monitor"
    ]

    def __init__(self, root, train, download=True):
        '''
        Params
            - download: Specify True to download VOC2012 and SBD datasets, if not already downloaded.
        '''
        super(PascalVOCSegReader, self).__init__(root, None, None, None)
        self.train = train

        if download:
            maybe_download_voc(root)
        
        base_dir = os.path.join(root, "PASCAL_SBDAUG")
        if not os.path.exists(base_dir):
            # Merge VOC and SBD datasets and create auxiliary files
            try:
                create_pascal_voc_aug(root)
            except (Exception, KeyboardInterrupt) as e:
                # Dataset creation fail for some reason...
                shutil.rmtree(base_dir)
                raise e
                
        if train:
            name_path = os.path.join(base_dir, 'trainaug.txt')
            class_map_dir = os.path.join(base_dir, 'trainaug_class_map')
        else:
            name_path = os.path.join(base_dir, 'val.txt')
            class_map_dir = os.path.join(base_dir, 'val_class_map')
        
        # Read files
        name_list = list(np.loadtxt(name_path, dtype='str'))
        self.images = [os.path.join(base_dir, "raw_images", n + ".jpg") for n in name_list]
        self.targets = [os.path.join(base_dir, "annotations", n + ".npy") for n in name_list]
        
        # Given a class idx (1-20), self.class_map gives the list of images that contain
        # this class idx
        self.class_map = {}
        for c in range(1, 21):
            class_map_path = os.path.join(class_map_dir, str(c) + ".txt")
            class_name_list = list(np.loadtxt(class_map_path, dtype='str'))
            # Map name to indices
            class_idx_list = [name_list.index(n) for n in class_name_list]
            self.class_map[c] = class_idx_list

    def __len__(self):
        return len(self.images)

    def get_class_map(self, class_id):
        """
        Given a class label id (e.g., 2), return a list of all images in
        the dataset containing at least one pixel of the class.

        Parameters:
            - class_id: an integer representing class

        Return:
            - a list of all images in the dataset containing at least one pixel of the class
        """
        return deepcopy(self.class_map[class_id])
    
    def get_label_range(self):
        return [i + 1 for i in range(20)]

    def __getitem__(self, idx):
        # For both SBD and VOC2012, images are stored as .jpg
        img = Image.open(self.images[idx]).convert("RGB")

        target_np = np.load(self.targets[idx])

        return img, torch.tensor(target_np).long()

def get_train_set(cfg):
    # Note: previous works including FCN (https://arxiv.org/pdf/1411.4038.pdf)
    # or OSLSM (https://arxiv.org/pdf/1709.03410.pdf) use SBD annotations.
    # Here we follow the convention and use augmented notations from SBD
    ds = PascalVOCSegReader(utils.get_dataset_root(), True, download=True)
    return base_set(ds, "train", cfg)

def get_val_set(cfg):
    ds = PascalVOCSegReader(utils.get_dataset_root(), False, download=True)
    return base_set(ds, "test", cfg)
