import numpy as np
import torch
import torch.nn as nn

class backbone_base(nn.Module):
    '''
    Base class of backbone
    '''
    def __init__(self, cfg):
        super(backbone_base, self).__init__()
        self._cfg = cfg
        self.feature_shape_ = None
    
    def get_feature_tensor_shape(self, device_ = None):
        if self.feature_shape_ is not None:
            # Cached
            return self.feature_shape_
        self.eval()
        with torch.no_grad():
            input_dim = self._cfg.input_dim
            input_dim = (1,) + input_dim # batch of 1 image
            dummy_tensor = torch.rand(input_dim, device = device_)
            output = self.forward(dummy_tensor)
            assert len(output.shape) == 4, "Expect feature map to be BxCxHxW!"
            return output.shape

    def forward(self, x):
        raise NotImplementedError