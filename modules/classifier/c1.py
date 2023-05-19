import torch
import torch.nn as nn
import torch.nn.functional as F

class SameConvBNReLU(nn.ModuleDict):
    """
    Same padded conv2d with optional bn and relu
    """

    def __init__(self, in_channel, out_channel, kernel_size=3, stride=1,
                 has_bn=True, has_relu=True):
        super(SameConvBNReLU, self).__init__()
        self.conv = nn.Conv2d(
            in_channels=in_channel,
            out_channels=out_channel,
            kernel_size=kernel_size,
            stride=stride,
            padding=kernel_size // 2,
            bias=(not has_bn),
        )
        nn.init.kaiming_normal_(self.conv.weight, mode='fan_out')
        if self.conv.bias is not None:
            nn.init.zeros_(self.conv.bias)
        self.bn = nn.BatchNorm2d(out_channel) if has_bn else nn.Identity()
        self.relu = nn.ReLU() if has_relu else nn.Identity()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class c1(nn.Module):
    def __init__(self, cfg, feature_shape, num_classes, intermediate_channels_=512):
        super(c1, self).__init__()
        assert len(feature_shape) == 4, "Expect B*C*H*W"
        self.num_classes = num_classes
        self.prv_channels = feature_shape[1]
        self.conv = SameConvBNReLU(self.prv_channels, intermediate_channels_)
        self.final_conv = nn.Conv2d(intermediate_channels_, self.num_classes, 1, 1)

    def forward(self, x, size_=None):
        x = self.conv(x)
        x = self.final_conv(x)
        assert size_ is not None
        x = F.interpolate(x, size = size_, mode = 'bilinear', align_corners=False)
        return x