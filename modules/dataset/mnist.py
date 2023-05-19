import os
from torchvision import datasets, transforms

import utils
from .baseset import base_set

MNIST_PATH = os.path.join(utils.get_dataset_root(), "mnist")

download_ds = False if os.path.exists(MNIST_PATH) else True

def get_train_set(cfg):
    ds = datasets.MNIST(MNIST_PATH, train = True, download = download_ds)
    return base_set(ds, "train", cfg)

def get_val_set(cfg):
    ds =  datasets.MNIST(MNIST_PATH, train = False, download = download_ds)
    return base_set(ds, "test", cfg)