import torch
from torch.autograd import Variable
import torch.nn as nn
import math
import numpy as np
import torch.nn.functional as F

class euclidean(nn.Module):
    def __init__(self, cfg, feature_shape, outdim):
        super(cos, self).__init__()
        assert len(feature_shape) == 4, "Expect B*C*H*W"
        assert feature_shape[0] == 1, "Expect batch to be 1"
        indim = np.prod(feature_shape)
        self.L = nn.Linear( indim, outdim, bias = False)  

        self.scale_factor = 30

    def forward(self, x, scale_factor=None):
        x = torch.flatten(x, start_dim = 1)
        x_norm = torch.norm(x, p=2, dim =1).unsqueeze(1).expand_as(x)
        x_normalized = x.div(x_norm+ 1e-5)
        L_norm = torch.norm(self.L.weight.data, p=2, dim =1).unsqueeze(1).expand_as(self.L.weight.data)
        self.L.weight.data = self.L.weight.data.div(L_norm + 1e-5)
        cos_dist = self.L(x_normalized)

        if scale_factor is not None:
            return scale_factor * cos_dist
        else:
            return self.scale_factor * cos_dist
