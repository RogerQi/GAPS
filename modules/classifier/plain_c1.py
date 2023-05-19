import torch
import torch.nn as nn
import torch.nn.functional as F

class plain_c1(nn.Module):
    def __init__(self, cfg, feature_shape, num_classes):
        super(plain_c1, self).__init__()
        self.clf_conv = nn.Conv2d(
            feature_shape[1], num_classes, kernel_size=1, stride=1, padding=1, bias=True
        )

    def forward(self, x, size_=None):
        x = self.clf_conv(x)
        assert size_ is not None
        x = F.interpolate(x, size = size_, mode = 'bilinear', align_corners=False)
        return x
