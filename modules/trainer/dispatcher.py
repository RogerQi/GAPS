import torch
import torch.nn as nn
import torch.nn.functional as F

def dispatcher(cfg):
    task_name = cfg.task
    assert task_name != "none"
    if task_name == "classification":
        from .clf_trainer import clf_trainer as clf_trainer_fn
        return clf_trainer_fn
    elif task_name == "semantic_segmentation":
        from .seg_trainer import seg_trainer as seg_trainer_fn
        return seg_trainer_fn
    elif task_name == "GIFS":
        from .GIFS_seg_trainer import GIFS_seg_trainer as GIFS_seg_trainer_fn
        return GIFS_seg_trainer_fn
    elif task_name == "sequential_GIFS":
        from .sequential_GIFS_seg_trainer import sequential_GIFS_seg_trainer as sequential_GIFS_seg_trainer_fn
        return sequential_GIFS_seg_trainer_fn
    elif task_name == "few_shot_incremental":
        from .fs_incremental_trainer import fs_incremental_trainer as fs_incremental_trainer_fn
        return fs_incremental_trainer_fn
    elif task_name == "non_few_shot_incremental":
        from .non_fs_incremental_trainer import non_fs_incremental_trainer as non_fs_incremental_trainer_fn
        return non_fs_incremental_trainer_fn
    elif task_name == "live_continual_seg":
        from .live_continual_seg_trainer import live_continual_seg_trainer as live_con_seg_trainer_fn
        return live_con_seg_trainer_fn
    else:
        raise NotImplementedError