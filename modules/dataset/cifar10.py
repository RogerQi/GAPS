import sys
import os
import torch
from torchvision import datasets, transforms

import utils
from .baseset import base_set

CIFAR10_PATH = os.path.join(utils.get_dataset_root(), 'cifar10')

download_ds = False if os.path.exists(CIFAR10_PATH) else True

def get_train_set(cfg):
    ds = datasets.CIFAR10(CIFAR10_PATH, train = True, download = download_ds)
    return base_set(ds, "train", cfg)

def get_val_set(cfg):
    ds =  datasets.CIFAR10(CIFAR10_PATH, train = False, download = download_ds)
    return base_set(ds, "test", cfg)