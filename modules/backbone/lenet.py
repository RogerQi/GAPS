import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbone_base import backbone_base

class net(backbone_base):
    '''
    Implementation of LeNet from Pytorch official.

    Good for testing if the pipeline is working.
    '''
    def __init__(self, cfg):
        super(net, self).__init__(cfg)
        self.pooling = cfg.BACKBONE.pooling
        in_channel = cfg.input_dim[0]
        self.conv1 = nn.Conv2d(in_channel, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        if self.pooling:
            x = F.max_pool2d(x, 2)
        return x