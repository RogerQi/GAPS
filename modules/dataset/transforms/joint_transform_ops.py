import random
import numpy as np
import torch
import torchvision
import torchvision.transforms.functional as tr_F
from .transforms_registry import registry

joint_transforms_registry = registry()

######################
# Crop
######################
@joint_transforms_registry.register
class joint_random_crop:
    def __init__(self, transforms_cfg):
        self.output_H, self.output_W = transforms_cfg.TRANSFORMS_DETAILS.crop_size
    
    def __call__(self, img, target):
        assert img.shape[-2:] == target.shape[-2:]
        assert len(img.shape) == 3, "Only C x H x W images are supported"
        H, W = img.shape[-2:]
        if H <= self.output_H or W <= self.output_W:
            # Pad and crop from zero
            temp_size = (max(H, self.output_H), max(W, self.output_W))
            temp_shape = img.shape[:-2] + temp_size
            img_bg = torch.zeros(temp_shape, dtype = img.dtype, device = img.device)
            img_bg[:, 0:H, 0:W] = img
            img = img_bg
            if len(target.shape) == 2:
                target_bg = torch.zeros(temp_size, dtype = target.dtype, device = target.device)
                target_bg[0:H, 0:W] = target
                target = target_bg
            else:
                assert len(target.shape) == 3, "This block supports C x H x W binary labels"
                C = target.shape[0]
                target_bg = torch.zeros((C,) + temp_size, dtype = target.dtype, device = target.device)
                target_bg[:, 0:H, 0:W] = target
                target = target_bg
            i = j = 0
        else:
            i = random.randint(0, H - self.output_H)
            j = random.randint(0, W - self.output_W)
        if len(target.shape) == 2:
            return (img[:, i:i + self.output_H, j:j + self.output_W], target[i:i + self.output_H, j:j + self.output_W])
        else:
            return (img[:, i:i + self.output_H, j:j + self.output_W], target[:, i:i + self.output_H, j:j + self.output_W])

@joint_transforms_registry.register
class joint_random_scale_crop:
    def __init__(self, transforms_cfg):
        self.output_H, self.output_W = transforms_cfg.TRANSFORMS_DETAILS.crop_size

    def __call__(self, img, target):
        assert img.shape[-2:] == target.shape[-2:], "image and label map size mismatched"
        img_H = img.shape[-2]
        img_W = img.shape[-1]
        # Random scale
        scale = np.random.uniform(0.5, 2)
        target_H, target_W = int(img_H * scale), int(img_W * scale)
        img = tr_F.resize(img, (target_H, target_W))
        assert len(target.shape) == 2
        target = target.view((1,) + target.shape)
        target = tr_F.resize(target, (target_H, target_W), interpolation=torchvision.transforms.InterpolationMode.NEAREST)
        # Random crop
        H_padding = int(max(0, np.ceil((self.output_H - target_H) / 2)))
        W_padding = int(max(0, np.ceil((self.output_W - target_W) / 2)))
        img = tr_F.pad(img, (W_padding, H_padding), 0, 'constant')
        target = tr_F.pad(target, (W_padding, H_padding), 0, 'constant')
        # Restore target label map to (H, W)
        target = target.view(target.shape[1:])
        start_x = random.randint(0, target.shape[1] - self.output_W)
        start_y = random.randint(0, target.shape[0] - self.output_H)
        img = img[:,start_y:start_y+self.output_H,start_x:start_x+self.output_W]
        target = target[start_y:start_y+self.output_H,start_x:start_x+self.output_W]
        return (img, target)

@joint_transforms_registry.register
class joint_keep_ratio_resize:
    """Resize such that the longest edge is crop_size,
        followed by center crop with 0 padding on the short edge.
    """
    def __init__(self, transforms_cfg):
        self.output_H, self.output_W = transforms_cfg.TRANSFORMS_DETAILS.crop_size
        assert self.output_H == self.output_W

    def __call__(self, img, target):
        assert img.shape[-2:] == target.shape[-2:]
        img_H = img.shape[-2]
        img_W = img.shape[-1]
        scale = self.output_W / float(max(img_H, img_W))
        target_H, target_W = int(img_H * scale), int(img_W * scale)
        img = tr_F.resize(img, (target_H, target_W))
        if len(target.shape) == 2:
            # HxW?
            target = target.reshape((1,) + target.shape)
            target = tr_F.resize(target, (target_H, target_W), interpolation=torchvision.transforms.InterpolationMode.NEAREST)
            target = target.reshape(target.shape[1:])
        else:
            assert len(target.shape) == 3
            target = tr_F.resize(target, (target_H, target_W), interpolation=torchvision.transforms.InterpolationMode.NEAREST)
        return (img, target)

@joint_transforms_registry.register
class joint_center_crop:
    def __init__(self, transforms_cfg):
        self.output_H, self.output_W = transforms_cfg.TRANSFORMS_DETAILS.crop_size
        assert self.output_H == self.output_W

    def __call__(self, img, target):
        assert img.shape[-2:] == target.shape[-2:]
        img = tr_F.center_crop(img, (self.output_H, self.output_W))
        target = tr_F.center_crop(img, (self.output_H, self.output_W))
        return (img, target)

@joint_transforms_registry.register
class joint_resize_center_crop:
    """First resize the image's shortest edge to crop_size. Then perform center crop

    This is consistent with various incremental segmentation papers' validation implementations
        - https://github.com/clovaai/SSUL/blob/main/main.py#L114
        - https://github.com/arthurdouillard/CVPR2021_PLOP/blob/main/run.py#L51
    """
    def __init__(self, transforms_cfg):
        self.output_H, self.output_W = transforms_cfg.TRANSFORMS_DETAILS.crop_size
        assert self.output_H == self.output_W

    def __call__(self, img, target):
        assert img.shape[-2:] == target.shape[-2:]
        img = tr_F.resize(img, (self.output_H, self.output_W))
        if len(target.shape) == 2:
            # HxW?
            target = target.reshape((1,) + target.shape)
            target = tr_F.resize(target, (self.output_H, self.output_W), interpolation=torchvision.transforms.InterpolationMode.NEAREST)
            target = tr_F.center_crop(target, (self.output_H, self.output_W))
            target = target.reshape(target.shape[1:])
        else:
            assert len(target.shape) == 3
            target = tr_F.resize(target, (self.output_H, self.output_W), interpolation=torchvision.transforms.InterpolationMode.NEAREST)
            target = tr_F.center_crop(target, (self.output_H, self.output_W))
        return (img, target)

@joint_transforms_registry.register
class joint_random_horizontal_flip:
    # The default setting is p=0.5 in torchvision
    # https://pytorch.org/vision/stable/transforms.html#torchvision.transforms.RandomHorizontalFlip
    def __init__(self, transforms_cfg):
        pass

    def __call__(self, img, target):
        if torch.rand(1) < 0.5:
            img = tr_F.hflip(img)
            target = tr_F.hflip(target)
        return (img, target)
