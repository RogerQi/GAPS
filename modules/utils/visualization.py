import torch
import numpy as np
from PIL import Image
import imgviz
import matplotlib.pyplot as plt

def norm_tensor_to_np(cfg, arr):
    """
    Parameters:
        - cfg: config node
        - arr: normalized floating point torch.Tensor of shape (3, H, W)
    """
    assert isinstance(arr, torch.Tensor)
    assert arr.shape[0] == 3
    assert len(arr) == 3
    ori_rgb_np = np.array(arr.permute((1, 2, 0)).cpu()) # H x W x 3
    if 'normalize' in cfg.DATASET.TRANSFORM.TEST.transforms:
        rgb_mean = cfg.DATASET.TRANSFORM.TEST.TRANSFORMS_DETAILS.NORMALIZE.mean
        rgb_sd = cfg.DATASET.TRANSFORM.TEST.TRANSFORMS_DETAILS.NORMALIZE.sd
        ori_rgb_np = (ori_rgb_np * rgb_sd) + rgb_mean
    assert ori_rgb_np.max() <= 1.1, "Max is {}".format(ori_rgb_np.max())
    ori_rgb_np[ori_rgb_np >= 1] = 1
    arr = (ori_rgb_np * 255).astype(np.uint8)
    return arr

def save_to_disk(cfg, arr, path):
    if isinstance(arr, torch.Tensor) and len(arr.shape) == 3 and arr.shape[0] == 3:
        # TODO: check tensor type (float) to denormalize.
        arr = norm_tensor_to_np(cfg, arr)
    elif isinstance(arr, torch.Tensor) and len(arr.shape) == 2:
        arr = arr.cpu().detach().numpy()
        if arr.dtype == np.dtype('int64'):
            assert arr.max() < 256
            arr = arr.astype(np.uint8)
    else:
        raise NotImplementedError
    im = Image.fromarray(arr)
    im.save(path)

# A generalized imshow helper function which supports displaying (CxHxW) tensor
def generalized_imshow(cfg, arr):
    '''
    Parameters
        - cfg: root yacs node of the YAML file
        - arr:
            normalized numpy array
            unnormalized pytorch tensor
    '''
    if isinstance(arr, torch.Tensor) and arr.shape[0] == 3:
        # TODO: check tensor type (float) to denormalize.
        arr = norm_tensor_to_np(cfg, arr)
    plt.margins(x=0)
    plt.axis('off')
    plt.imshow(arr)
    plt.show()

def visualize_segmentation(cfg, img, label_np, class_names_list):
    """
    img: 3 x H x W. range 0-1
    label: H x W. range 0-num_classes
    """
    if class_names_list is not None:
        assert label_np.max() < len(class_names_list)
    ori_img = norm_tensor_to_np(cfg, img)
    rgb_img_w_lbl = imgviz.label2rgb(
        label=label_np, image=imgviz.asgray(ori_img), label_names=class_names_list, loc="rb"
    )

    return rgb_img_w_lbl
