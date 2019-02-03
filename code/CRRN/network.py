import sys
sys.path.append('..')
import torch
import torch.nn as nn
from base_model import vgg


class GradientInferenceNetwork(nn.Module):
    def __init__(self):
        super(GradientInferenceNetwork, self).__init__()

    def forward(self, x):
        return x


class ImageInferenceNetwork(nn.Module):
    backbone_list = ['vgg16', 'vgg16_bn']

    def __init__(self, backbone_type=None):
        super(ImageInferenceNetwork, self).__init__()
        if backbone_type not in self.backbone_list:
            raise ValueError("not support backbone type: {}".format(backbone_type))
        if backbone_type == 'vgg16':
            self.backbone = vgg.vgg16(pretrained=False, is_backbone=True)
        elif backbone_type == 'vgg16_bn':
            self.backbone = vgg.vgg16_bn(pretrained=False, is_backbone=True)

    def forward(self, x):
        x = self.backbone(x)
        return x


if __name__ == "__main__":
    GiN = GradientInferenceNetwork()
    IiN = ImageInferenceNetwork(backbone_type='vgg16')
    print(IiN)
