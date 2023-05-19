import torch
import torch.nn as nn
import torch.nn.functional as F

def dispatcher(cfg):
    loss_name = cfg.LOSS.loss
    assert loss_name != "none"
    if loss_name == "cross_entropy":
        from .loss import cross_entropy
        return cross_entropy(cfg)
    elif loss_name == "binary_cross_entropy":
        return nn.BCEWithLogitsLoss()
    elif loss_name == "semantic_nllloss":
        from .loss import semantic_segmentation_nllloss
        return semantic_segmentation_nllloss(cfg)
    else:
        raise NotImplementedError