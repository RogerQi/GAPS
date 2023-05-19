import os
import yaml
from copy import deepcopy

from yacs.config import CfgNode as CN

# ----------------------------
# | Start Default Config
# ----------------------------

#######################
# Root Config Node
#######################
_C = CN()
_C.BASE_YAML = "__BASE__YAML__"
_C.name = "Experiment Name"
_C.seed = 1221
_C.task = "classification"
_C.num_classes = -1
_C.meta_training_num_classes = -1
_C.meta_testing_num_classes = -1
_C.input_dim = (3, 32, 32)
_C.save_model = False

#######################
# DL System Setting
#######################
_C.SYSTEM = CN()
_C.SYSTEM.pin_memory = True
_C.SYSTEM.num_gpus = 1		# Number of GPUs to use
_C.SYSTEM.num_workers = 4	# Number of CPU workers for errands

#######################
# Backbone
#######################
_C.BACKBONE = CN()
_C.BACKBONE.network = "none"
_C.BACKBONE.use_pretrained = False
_C.BACKBONE.pretrained_path = "none"
_C.BACKBONE.forward_need_label = False
_C.BACKBONE.pooling = False

#######################
# Classification Layer
#######################
_C.CLASSIFIER = CN()
_C.CLASSIFIER.classifier = "none"
_C.CLASSIFIER.factor = 0
_C.CLASSIFIER.FC = CN()
_C.CLASSIFIER.FC.hidden_layers = (1024,)
_C.CLASSIFIER.FC.bias = False

#######################
# Network (ignore backbone & classification)
#######################
_C.NETWORK = CN()
_C.NETWORK.network = "none"

#######################
# Loss
#######################
_C.LOSS = CN()
_C.LOSS.loss = "none"
_C.LOSS.loss_factor = 1.

#######################
# Training Settings (used for both usual training and meta-trainint)
#######################
_C.TRAIN = CN()
_C.TRAIN.log_interval = 10
_C.TRAIN.batch_size = 64
_C.TRAIN.initial_lr = 0.01
_C.TRAIN.lr_scheduler = 'none'
_C.TRAIN.lr_max_epoch = -1
_C.TRAIN.step_down_gamma = 0.1
_C.TRAIN.step_down_on_epoch = []
_C.TRAIN.max_epochs = 100
_C.TRAIN.OPTIMIZER = CN()
_C.TRAIN.OPTIMIZER.type = 'SGD'
_C.TRAIN.OPTIMIZER.momentum = 0.9
_C.TRAIN.OPTIMIZER.weight_decay = 1e-4

#######################
# Meta-testing Train Settings
#######################
_C.META_TEST = CN()
_C.META_TEST.log_interval = 10
_C.META_TEST.shot = -1
_C.META_TEST.initial_lr = 0.01
_C.META_TEST.lr_scheduler = 'none'
_C.META_TEST.step_down_gamma = 0.1
_C.META_TEST.step_down_on_epoch = []
_C.META_TEST.max_epochs = 100
_C.META_TEST.OPTIMIZER = CN()
_C.META_TEST.OPTIMIZER.type = 'SGD'
_C.META_TEST.OPTIMIZER.momentum = 0.9
_C.META_TEST.OPTIMIZER.weight_decay = 1e-4

#######################
# Validation Settings
#######################
_C.VAL = CN()

#######################
# Test Settings
#######################
_C.TEST = CN()
_C.TEST.batch_size = 256

#######################
# Metric Settings
#######################
_C.METRIC = CN()
_C.METRIC.CLASSIFICATION = CN()
_C.METRIC.SEGMENTATION = CN()
_C.METRIC.SEGMENTATION.fg_only = False

#######################
# Dataset Settings
#######################
_C.DATASET = CN()
_C.DATASET.dataset = 'cifar10'
_C.DATASET.cache_all_data = False
_C.DATASET.NUMPY_READER = CN()
_C.DATASET.NUMPY_READER.train_data_npy_path = "/"
_C.DATASET.NUMPY_READER.train_label_npy_path = "/"
_C.DATASET.NUMPY_READER.test_data_npy_path = "/"
_C.DATASET.NUMPY_READER.test_label_npy_path = "/"
_C.DATASET.NUMPY_READER.mmap = False
_C.DATASET.PASCAL5i = CN()
_C.DATASET.PASCAL5i.folding = -1
_C.DATASET.PASCAL5i.disjoint = True # See https://github.com/fcdl94/MiB
_C.DATASET.COCO20i = CN()
_C.DATASET.COCO20i.folding = -1

#######################
# Transform Settings
#######################
_C.DATASET.TRANSFORM = CN()
_C.DATASET.TRANSFORM.TRAIN = CN()
_C.DATASET.TRANSFORM.TRAIN.transforms = ('none', )
_C.DATASET.TRANSFORM.TRAIN.joint_transforms = ('none', )
_C.DATASET.TRANSFORM.TRAIN.TRANSFORMS_DETAILS = CN()
_C.DATASET.TRANSFORM.TRAIN.TRANSFORMS_DETAILS.resize_size = (32, 32)
_C.DATASET.TRANSFORM.TRAIN.TRANSFORMS_DETAILS.crop_size = (32, 32)
_C.DATASET.TRANSFORM.TRAIN.TRANSFORMS_DETAILS.NORMALIZE = CN()
_C.DATASET.TRANSFORM.TRAIN.TRANSFORMS_DETAILS.NORMALIZE.mean = (0, 0, 0)
_C.DATASET.TRANSFORM.TRAIN.TRANSFORMS_DETAILS.NORMALIZE.sd = (1, 1, 1)
_C.DATASET.TRANSFORM.TRAIN.TRANSFORMS_DETAILS.RANDOM_RESIZE = CN()
# Duplicate transform for TEST
_C.DATASET.TRANSFORM.TEST = _C.DATASET.TRANSFORM.TRAIN.clone()

#######################
# Task-specific Settings
#######################
_C.TASK_SPECIFIC = CN()

#######################
# GIFS Reproduction: https://arxiv.org/pdf/2012.01415.pdf
#######################
_C.TASK_SPECIFIC.GIFS = CN()
_C.TASK_SPECIFIC.GIFS.num_shots = -1
_C.TASK_SPECIFIC.GIFS.fine_tuning = True
_C.TASK_SPECIFIC.GIFS.pseudo_base_label = False
_C.TASK_SPECIFIC.GIFS.ft_batch_size = 5
_C.TASK_SPECIFIC.GIFS.max_iter = 1000
_C.TASK_SPECIFIC.GIFS.backbone_lr = 1e-3
_C.TASK_SPECIFIC.GIFS.classifier_lr = 1e-3
_C.TASK_SPECIFIC.GIFS.feature_reg_lambda = 0.1
_C.TASK_SPECIFIC.GIFS.classifier_reg_lambda = 0.1
_C.TASK_SPECIFIC.GIFS.context_aware_sampling_prob = 0.0
_C.TASK_SPECIFIC.GIFS.sequential_dataset_num_classes = 1
_C.TASK_SPECIFIC.GIFS.baseset_type = 'random' # ('random', 'far', 'close', 'far_close')
_C.TASK_SPECIFIC.GIFS.probabilistic_synthesis_strat = 'vRFS' # ('vRFS', 'always', 'always_no', 'CAS', 'half')
_C.TASK_SPECIFIC.GIFS.num_runs = -1


# ---------------------------
# | End Default Config
# ---------------------------

BASE_KEY = "__BASE__YAML__"

def update_config_from_yaml(cfg, args):
    '''
    Update yacs config using yaml file
    '''
    loop_cnt = 0
    yaml_load_list = [args.cfg]
    cur_cfg_path = args.cfg
    with open(args.cfg) as f: cur_cfg = yaml.safe_load(f)
    while 'BASE_YAML' in cur_cfg and cur_cfg['BASE_YAML'] != BASE_KEY:
        cur_cfg_path = os.path.join(os.path.dirname(cur_cfg_path), cur_cfg['BASE_YAML'])
        with open(cur_cfg_path) as f:
            cur_cfg = yaml.safe_load(f)
        yaml_load_list.append(cur_cfg_path)
        loop_cnt += 1
        if loop_cnt > 1000:
            raise ValueError(f"Recursed over 1000 YAML. Did I get stuck in recursion? {yaml_load_list}")
    yaml_load_list = yaml_load_list[::-1] # reverse

    cfg.defrost()

    # Iteratively load YAML configs
    for cfg_path in yaml_load_list:
        cfg.merge_from_file(cfg_path)
    
    # Command line options have highest priorities
    if args.opts:
        # empty by default
        old_cfg = deepcopy(cfg)
        cfg.merge_from_list(args.opts)
        assert cfg != old_cfg, "Unused command line options provided!"

    cfg.freeze()

if __name__ == "__main__":
    # debug print
    print(_C)
