import random
import numpy as np
from PIL import Image, ImageFilter, ImageEnhance, ImageOps
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import ImageEnhance
from .transforms_registry import registry

my_transforms_registry = registry()

######################
# Crop
######################
@my_transforms_registry.register
def random_resized_crop(transforms_cfg):
    # TODO(roger): unify crop_size/resize_size and input_dim in config
    size = transforms_cfg.TRANSFORMS_DETAILS.crop_size
    op = transforms.RandomResizedCrop(size)
    return op

@my_transforms_registry.register
def random_crop(transforms_cfg):
    size = transforms_cfg.TRANSFORMS_DETAILS.crop_size
    op = transforms.RandomCrop(size, padding = 4)
    return op

@my_transforms_registry.register
def random_horizontal_flip(transforms_cfg):
    op = transforms.RandomHorizontalFlip(p = 0.5)
    return op

@my_transforms_registry.register
def center_crop(transforms_cfg):
    size = transforms_cfg.TRANSFORMS_DETAILS.crop_size
    op = transforms.CenterCrop(size)
    return op

@my_transforms_registry.register
def resize(transforms_cfg):
    size = transforms_cfg.TRANSFORMS_DETAILS.resize_size
    op = transforms.Resize(size, interpolation = Image.BILINEAR)
    return op

@my_transforms_registry.register
def resize_and_center_crop(transforms_cfg):
    """
    Resize to 1.15x image size and then center crop.

    Compared to naive center crop, this should preserve more foreground and benefits classification.

    The magic number 1.15 was obtained from https://github.com/wyharveychen/CloserLookFewShot/blob/master/data/datamgr.py
    """
    size = transforms_cfg.TRANSFORMS_DETAILS.resize_size
    assert len(size) == 2
    op = transforms.Resize((int(size[0] * 1.15), int(size[1] * 1.15)), interpolation = Image.BILINEAR)
    op2 = transforms.CenterCrop(size)
    def resize_and_center_crop_func(img):
        return op2(op(img))
    return resize_and_center_crop_func

######################
# Color
######################

@my_transforms_registry.register
def color_jitter(transform_cfg):
    op = transforms.ColorJitter(brightness=0.4, contrast=0.2, hue=0.2)
    return op

# Custom color jittering function from https://github.com/wyharveychen/CloserLookFewShot

class CustomImageJitter(object):
    def __init__(self, transformdict):
        transformtypedict=dict(
            Brightness=ImageEnhance.Brightness,
            Contrast=ImageEnhance.Contrast,
            Sharpness=ImageEnhance.Sharpness,
            Color=ImageEnhance.Color
        )
        self.transforms = [(transformtypedict[k], transformdict[k]) for k in transformdict]

    def __call__(self, img):
        out = img
        randtensor = torch.rand(len(self.transforms))

        for i, (transformer, alpha) in enumerate(self.transforms):
            r = alpha*(randtensor[i]*2.0 -1.0) + 1
            out = transformer(out).enhance(r).convert('RGB')

        return out

@my_transforms_registry.register
def fs_cls_color_jitter(transform_cfg):
    op = CustomImageJitter(dict(Brightness=0.4, Contrast=0.4, Color=0.4))
    return op


######################
# Geometric Transform
######################