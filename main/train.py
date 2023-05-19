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
    parser.add_argument('--resume', help = "resume training from checkpoint", required=False, default = "NA", type = str)
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

    print(cfg)
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
    print("Backbone output feature tensor shape: {}".format(feature_shape))
    post_processor = classifier.dispatcher(cfg, feature_shape)
    
    post_processor = post_processor.to(device)
    
    if cfg.BACKBONE.use_pretrained:
        weight_path = cfg.BACKBONE.pretrained_path
        print("Initializing backbone with pretrained weights from: {}".format(weight_path))
        pretrained_weight_dict = torch.load(weight_path)
        # Log missing/uncompatible keys
        if 'backbone' in pretrained_weight_dict:
            print(backbone_net.load_state_dict(pretrained_weight_dict['backbone'], strict=False))
        else:
            print(backbone_net.load_state_dict(pretrained_weight_dict, strict=False))

    criterion = loss.dispatcher(cfg)

    trainable_params = list(backbone_net.parameters()) + list(post_processor.parameters())

    if cfg.TRAIN.OPTIMIZER.type == "adadelta":
        optimizer = optim.Adadelta(trainable_params, lr = cfg.TRAIN.initial_lr,
                                    weight_decay = cfg.TRAIN.OPTIMIZER.weight_decay)
    elif cfg.TRAIN.OPTIMIZER.type == "SGD":
        optimizer = optim.SGD(trainable_params, lr = cfg.TRAIN.initial_lr, momentum = cfg.TRAIN.OPTIMIZER.momentum,
                                weight_decay = cfg.TRAIN.OPTIMIZER.weight_decay)
    elif cfg.TRAIN.OPTIMIZER.type == "ADAM":
        optimizer = optim.Adam(trainable_params, lr = cfg.TRAIN.initial_lr, betas = (0.9, 0.999),
                                weight_decay = cfg.TRAIN.OPTIMIZER.weight_decay)
    else:
        raise NotImplementedError("Got unsupported optimizer: {}".format(cfg.TRAIN.OPTIMIZER.type))

    # Prepare LR scheduler
    if cfg.TRAIN.lr_max_epoch != -1:
        max_epoch = cfg.TRAIN.lr_max_epoch
    else:
        max_epoch = cfg.TRAIN.max_epochs

    if cfg.TRAIN.lr_scheduler == "step_down":
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones = cfg.TRAIN.step_down_on_epoch,
                                                            gamma = cfg.TRAIN.step_down_gamma)
    elif cfg.TRAIN.lr_scheduler == "polynomial":
        def polynomial_schedule(epoch):
            # from https://arxiv.org/pdf/2012.01415.pdf
            return (1 - epoch / max_epoch)**0.9
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, polynomial_schedule)
    else:
        raise NotImplementedError("Got unsupported scheduler: {}".format(cfg.TRAIN.lr_scheduler))

    best_val_metric = 0

    trainer_func = trainer.dispatcher(cfg)
    my_trainer = trainer_func(cfg, backbone_net, post_processor, criterion, dataset_module, device)

    start_epoch = 1
    if args.resume != "NA":
        sub_str = args.resume[args.resume.index('epoch') + 5:]
        start_epoch = int(sub_str[:sub_str.index('_')]) + 1
        assert start_epoch <= cfg.TRAIN.max_epochs
        print("Resuming training from epoch {}".format(start_epoch))
        my_trainer.load_model(args.resume)

    # Tune LR scheduler
    for epoch in range(1, start_epoch):
        scheduler.step()

    for epoch in range(start_epoch, cfg.TRAIN.max_epochs + 1):
        start_cp = time.time()
        my_trainer.train_one(device, optimizer, epoch)
        scheduler.step()
        print("Training took {:.4f} seconds".format(time.time() - start_cp))
        start_cp = time.time()
        val_metric = my_trainer.val_one(device)
        print("Eval took {:.4f} seconds.".format(time.time() - start_cp))
        best_val_metric = val_metric
        if True:
            print("Epoch {} New Best Model w/ metric: {:.4f}".format(epoch, val_metric))
            best_val_metric = val_metric
            if cfg.save_model:
                best_model_path = "{0}_epoch{1}_{2:.4f}.pt".format(cfg.name, epoch, best_val_metric)
                print("Saving model to {}".format(best_model_path))
                my_trainer.save_model(best_model_path)
                
        print("===================================\n")

    if cfg.save_model:
        final_name = "{0}_final.pt".format(cfg.name)
        my_trainer.save_model(final_name)

if __name__ == '__main__':
    main()
