import __init_lib_path
from config_guard import cfg, update_config_from_yaml
import dataset
import backbone
import classifier
import loss
import trainer
import utils

import time
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

def parse_args():
    parser = argparse.ArgumentParser(description = "Roger's Deep Learning Playground")
    parser.add_argument('--cfg', help = "specify particular yaml configuration to use", required = True,
        default = "configs/mnist_torch_official.yaml", type = str)
    parser.add_argument('--load', help="specify saved checkpoint to evaluate", required=True, type=str)
    parser.add_argument('--jit_trace', help='Trace and serialize trained network. Overwrite other options', action='store_true')
    parser.add_argument('--webcam', help='real-time evaluate using default webcam', action='store_true')
    parser.add_argument('--visfreq', help="visualize results for every n examples in test set",
        required=False, default=99999999999, type=int)
    parser.add_argument("--opts", help="Command line options to overwrite configs", default=[], nargs=argparse.REMAINDER)
    args = parser.parse_args()

    return args

def main():
    # --------------------------
    # | Initial set up
    # |  1. Parse arg and yaml
    # |  2. Set device
    # |  3. Set seed
    # --------------------------
    args = parse_args()
    update_config_from_yaml(cfg, args)
    device = utils.guess_device()

    torch.manual_seed(cfg.seed)

    # --------------------------
    # | Prepare datasets
    # --------------------------
    dataset_module = dataset.dataset_dispatcher(cfg)

    # --------------------------
    # | Get ready to learn
    # |  1. Prepare network and loss
    # |  2. Prepare optimizer
    # |  3. Set learning rate
    # --------------------------
    backbone_net = backbone.dispatcher(cfg)
    backbone_net = backbone_net(cfg).to(device)
    feature_shape = backbone_net.get_feature_tensor_shape(device)
    print("Flatten feature length: {}".format(feature_shape))
    post_processor = classifier.dispatcher(cfg, feature_shape)
    
    post_processor = post_processor.to(device)

    criterion = loss.dispatcher(cfg)

    trainer_func = trainer.dispatcher(cfg)
    my_trainer = trainer_func(cfg, backbone_net, post_processor, criterion, dataset_module, device)

    print("Initializing backbone with trained weights from: {}".format(args.load))
    my_trainer.load_model(args.load)

    if args.jit_trace:
        trained_weight_path = args.load
        assert trained_weight_path.endswith('.pt')
        my_trainer.trace_model(trained_weight_path.replace('.pt', '_traced.pt'))
    elif args.webcam:
        my_trainer.live_run(device)
    else:
       val_metric = my_trainer.test_one(device)
       print(val_metric)

if __name__ == '__main__':
    main()
