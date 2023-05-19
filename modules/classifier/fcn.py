import torch
import torch.nn as nn
import torch.nn.functional as F

class fcn32s(nn.Module):

    def __init__(self, cfg, feature_shape, num_classes):
        super().__init__()
        self.num_classes = num_classes
        # fc6
        self.fc6 = nn.Conv2d(512, 4096, 7)
        self.relu6 = nn.ReLU(inplace=True)
        self.drop6 = nn.Dropout2d()

        # fc7
        self.fc7 = nn.Conv2d(4096, 4096, 1)
        self.relu7 = nn.ReLU(inplace=True)
        self.drop7 = nn.Dropout2d()

        self.score_fr = nn.Conv2d(4096, self.num_classes, 1)
        if False:
            # In the original FCN implementation, a Transpose Conv was used. The weights
            # of the Transpose Conv was initialized such that it mimics the behavior of
            # bilinear interpolation

            # In our implementation, we follow more recent work and deprecate this approach.
            # The code is left here just for reference.
            self.upscore = nn.ConvTranspose2d(self.num_classes, self.num_classes, 64, stride=32,
                                          bias=False)

    def forward(self, x, size_ = None):
        x = self.relu6(self.fc6(x))
        x = self.drop6(x)

        x = self.relu7(self.fc7(x))
        x = self.drop7(x)

        x = self.score_fr(x)

        if True:
            x = F.interpolate(x, size = size_, mode = 'bilinear', align_corners=False)
        else:
            x = self.upscore(x)
            x = x[:, :, 19:19 + size_[0], 19:19 + size_[1]].contiguous()

        return x
