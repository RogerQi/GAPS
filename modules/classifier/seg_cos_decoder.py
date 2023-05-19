import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class pixel_classifier(nn.Module):
    def __init__(self, in_channel, num_classes):
        super().__init__()
        self.num_classes = num_classes
        self.class_mat = nn.Conv2d(in_channel, self.num_classes, 1, bias = False)
        self.scale_factor = 10
    
    def forward(self, x, scale_factor=None):
        '''
        x: (B, in_channel, H, W)
        '''
        # x_norm: (B, in_channel, H, W) where x_norm[i, :, H, W] is the norm of
        # x[i, :, H, W]. That is, x/x_norm yields normalized value along channel axis
        x_norm = torch.norm(x, p=2, dim=1).unsqueeze(1).expand_as(x)
        x_normalized = x.div(x_norm + 1e-10) # avoid div by zero
        class_mat_norm = torch.norm(self.class_mat.weight.data, p=2, dim=1).unsqueeze(1).expand_as(self.class_mat.weight.data)
        self.class_mat.weight.data = self.class_mat.weight.data.div(class_mat_norm + 1e-10)
        cos_dist = self.class_mat(x_normalized)
        if scale_factor is not None:
            return scale_factor * cos_dist
        else:
            return self.scale_factor * cos_dist

class seg_cos_decoder(nn.Module):
    # Treat segmentation as a classification problem for every spatial pixel
    def __init__(self, cfg, feature_shape, num_classes):
        super().__init__()
        self.num_classes = num_classes
        self.in_channels = feature_shape[1]

        self.head = nn.Sequential(
                    nn.Conv2d(self.in_channels, 256, 3, padding=1, bias=False),
                    nn.BatchNorm2d(256),
                    nn.ReLU(inplace=True),
                    pixel_classifier(256, self.num_classes)
                )

        # self.pixel_classifier = pixel_classifier(cfg, feature_shape, self.in_channels, self.num_classes)
    
    def forward(self, x, size_=None, scale_factor=None):
        x = self.head(x)
        assert size_ is not None
        x = F.interpolate(x, size = size_, mode = 'bilinear')
        return x
    
    def _add_head(self, output_channels):
        raise NotImplementedError
