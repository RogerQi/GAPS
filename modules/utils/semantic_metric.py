from typing import List
from cv2 import getValidDisparityROI
import numpy as np
import torch
import matplotlib.pyplot as plt

from IPython import embed

IGNORE_LABEL = -1
eps=1e-10

def get_valid_mask(gt_arr, fg_only, masked_class):
    valid_mask = (gt_arr != IGNORE_LABEL)
    if fg_only:
        if isinstance(gt_arr, torch.Tensor):
            valid_mask = torch.logical_and(valid_mask, gt_arr > 0)
        elif isinstance(gt_arr, np.ndarray):
            valid_mask = np.logical_and(valid_mask, gt_arr > 0)
        else:
            raise NotImplementedError
    if masked_class is not None:
        assert isinstance(masked_class, List)
        for masked_c in masked_class:
            if isinstance(gt_arr, torch.Tensor):
                valid_mask = torch.logical_and(valid_mask, gt_arr != masked_c)
            elif isinstance(gt_arr, np.ndarray):
                valid_mask = np.logical_and(valid_mask, gt_arr != masked_c)
            else:
                raise NotImplementedError
    return valid_mask

def compute_pixel_acc(pred, label, fg_only=True, masked_class=None):
    '''
    pred: BHW
    label: BHW
    '''
    assert pred.shape == label.shape
    valid_mask = get_valid_mask(label, fg_only, masked_class)
    pred = pred[valid_mask]
    label = label[valid_mask]
    correct_sum = (pred == label).sum()
    valid_sum = valid_mask.sum()
    acc = float(correct_sum) / (valid_sum + 1e-10)
    return acc, valid_sum

def compute_binary_metrics(pred, label, class_idx=1, fg_only=False, masked_class=None):
    valid_mask = get_valid_mask(label, fg_only, masked_class)
    pred = (pred == class_idx).astype(np.uint8)
    label = (label == class_idx).astype(np.uint8)
    pred = pred[valid_mask]
    label = label[valid_mask]
    tp = np.sum(np.logical_and(pred == 1, label == 1))
    tn = np.sum(np.logical_and(pred == 0, label == 0))
    fp = np.sum(np.logical_and(pred == 1, label == 0))
    fn = np.sum(np.logical_and(pred == 0, label == 1))

    return {
        'tp': tp,
        'tn': tn,
        'fp': fp,
        'fn': fn,
        'acc': (tp + tn) / (tp + tn + fp + fn + eps),
        'iou': tp / (tp + fp + fn + eps),
        'recall': tp / (tp + fn + eps),
        'precision': tp / (tp + fp + eps)
    }

def compute_iou(pred_map, label_map, num_classes, fg_only=False, masked_class=None):
    """
    Param
        - ignore_mask: set to True if there are targets to be ignored. Pixels whose value equal to 255
            are excluded from benchmarking.
    """
    pred_map = np.asarray(pred_map).copy()
    label_map = np.asarray(label_map).copy()

    assert pred_map.shape == label_map.shape

    valid_idx = get_valid_mask(label_map, fg_only, masked_class)
    pred_map = pred_map[valid_idx]
    label_map = label_map[valid_idx]

    # When computing intersection, all pixels that are not
    # in the intersection are masked with zeros.
    # So we add 1 to the existing mask so that background pixels can be computed
    pred_map += 1
    label_map += 1

    # Compute area intersection:
    intersection = pred_map * (pred_map == label_map)
    (area_intersection, _) = np.histogram(
        intersection, bins=num_classes, range=(1, num_classes))

    # Compute area union:
    (area_pred, _) = np.histogram(pred_map, bins=num_classes, range=(1, num_classes))
    (area_lab, _) = np.histogram(label_map, bins=num_classes, range=(1, num_classes))
    area_union = area_pred + area_lab - area_intersection

    if fg_only:
        # Remove first bg channel
        return np.sum(area_intersection[1:]) / (np.sum(area_union[1:]) + 1e-10)
    else:
        return np.sum(area_intersection) / np.sum(area_union)

def compute_iu(pred_map, label_map, num_classes, fg_only=False, masked_class=None):
    """
    Param
        - ignore_mask: set to True if there are targets to be ignored. Pixels whose value equal to 255
            are excluded from benchmarking.
    """
    pred_map = np.asarray(pred_map).copy()
    label_map = np.asarray(label_map).copy()

    assert pred_map.shape == label_map.shape

    valid_idx = get_valid_mask(label_map, fg_only, masked_class)
    pred_map = pred_map[valid_idx]
    label_map = label_map[valid_idx]

    # When computing intersection, all pixels that are not
    # in the intersection are masked with zeros.
    # So we add 1 to the existing mask so that background pixels can be computed
    pred_map += 1
    label_map += 1

    # Compute area intersection:
    intersection = pred_map * (pred_map == label_map)
    (area_intersection, _) = np.histogram(
        intersection, bins=num_classes, range=(1, num_classes))

    # Compute area union:
    (area_pred, _) = np.histogram(pred_map, bins=num_classes, range=(1, num_classes))
    (area_lab, _) = np.histogram(label_map, bins=num_classes, range=(1, num_classes))
    area_union = area_pred + area_lab - area_intersection

    if fg_only:
        return (area_intersection[1:], area_union[1:])
    else:
        return (area_intersection, area_union)
