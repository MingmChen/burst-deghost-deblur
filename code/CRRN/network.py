import sys
from collections import OrderedDict
import torch
import torch.nn as nn
sys.path.append('..')
import base_model.nn_module as M


class GradientInferenceNetwork(nn.Module):
    def __init__(self):
        super(GradientInferenceNetwork, self).__init__()

    def forward(self, x):
        return x


class ReductionModuleA(nn.Module):
    def __init__(self, in_channels, out_channels, init_type="xavier", use_batchnorm=True):
        super(ReductionModuleA, self).__init__()
        self.init_type = init_type
        self.activation = nn.ReLU(inplace=True)
        self.bn = use_batchnorm
        self.branch0 = M.conv2d_block(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    init_type=self.init_type,
                    activation=self.activation,
                    use_batchnorm=self.bn
                )
        self.branch1 = nn.Sequential(
                M.conv2d_block(
                    in_channels=in_channels,
                    out_channels=out_channels//2,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    init_type=self.init_type,
                    activation=self.activation,
                    use_batchnorm=self.bn
                    ),
                M.conv2d_block(
                    in_channels=out_channels//2,
                    out_channels=out_channels//2,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    init_type=self.init_type,
                    activation=self.activation,
                    use_batchnorm=self.bn
                    ),
                M.conv2d_block(
                    in_channels=out_channels//2,
                    out_channels=out_channels,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    init_type=self.init_type,
                    activation=self.activation,
                    use_batchnorm=self.bn
                    )
                )
        self.branch2 = nn.Sequential(
                M.conv2d_block(
                    in_channels=in_channels,
                    out_channels=out_channels//2,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    init_type=self.init_type,
                    activation=self.activation,
                    use_batchnorm=self.bn
                    ),
                M.conv2d_block(
                    in_channels=out_channels//2,
                    out_channels=out_channels,
                    kernel_size=7,
                    stride=1,
                    padding=3,
                    init_type=self.init_type,
                    activation=self.activation,
                    use_batchnorm=self.bn
                    )
                )

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        output = torch.cat([x0, x1, x2], dim=1)
        return output


class ReductionModuleB(nn.Module):
    def __init__(self, in_channels, out_channels, init_type="xavier", use_batchnorm=True):
        super(ReductionModuleB, self).__init__()
        self.init_type = init_type
        self.activation = nn.ReLU(inplace=True)
        self.bn = use_batchnorm
        self.branch0 = nn.Sequential(
                M.conv2d_block(
                    in_channels=in_channels,
                    out_channels=out_channels//2,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    init_type=self.init_type,
                    activation=self.activation,
                    use_batchnorm=self.bn
                    ),
                M.conv2d_block(
                    in_channels=out_channels//2,
                    out_channels=out_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    init_type=self.init_type,
                    activation=self.activation,
                    use_batchnorm=self.bn
                    )
                )
        self.branch1 = nn.Sequential(
                M.conv2d_block(
                    in_channels=in_channels,
                    out_channels=out_channels//2,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    init_type=self.init_type,
                    activation=self.activation,
                    use_batchnorm=self.bn
                    ),
                M.conv2d_block(
                    in_channels=out_channels//2,
                    out_channels=out_channels//2,
                    kernel_size=(1, 7),
                    stride=1,
                    padding=(0, 3),
                    init_type=self.init_type,
                    activation=self.activation,
                    use_batchnorm=self.bn
                    ),
                M.conv2d_block(
                    in_channels=out_channels//2,
                    out_channels=out_channels//2,
                    kernel_size=(7, 1),
                    stride=1,
                    padding=(3, 0),
                    init_type=self.init_type,
                    activation=self.activation,
                    use_batchnorm=self.bn
                    ),
                M.conv2d_block(
                    in_channels=out_channels//2,
                    out_channels=out_channels,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    init_type=self.init_type,
                    activation=self.activation,
                    use_batchnorm=self.bn
                    )
                )
        self.branch2 = nn.Sequential(
                M.conv2d_block(
                    in_channels=in_channels,
                    out_channels=out_channels//2,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    init_type=self.init_type,
                    activation=self.activation,
                    use_batchnorm=self.bn
                    ),
                M.conv2d_block(
                    in_channels=out_channels//2,
                    out_channels=out_channels,
                    kernel_size=7,
                    stride=1,
                    padding=3,
                    init_type=self.init_type,
                    activation=self.activation,
                    use_batchnorm=self.bn
                    )
                )

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        output = torch.cat([x0, x1, x2], dim=1)
        return output


class InceptionModule(nn.Module):
    def __init__(self):
        super(InceptionModule, self).__init__()

    def forward(self, x):
        return x


class ImageInferenceNetwork(nn.Module):
    backbone_list = ['vgg16', 'vgg16_bn']
    vgg_cfg = {
        'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
        'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
        'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
        'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
    }

    def __init__(self, backbone_type=None, use_batchnorm=True, init_type="xavier"):
        super(ImageInferenceNetwork, self).__init__()
        if backbone_type not in self.backbone_list:
            raise ValueError(
                "not support backbone type: {}".format(backbone_type))

        self.bn = use_batchnorm
        self.maxpool = nn.MaxPool2d(kernel_size=2)
        self.activation = nn.ReLU(inplace=True)
        self.init_type = init_type
        self.conv_count = 0
        self.mp_count = 0

        if backbone_type == 'vgg16' or backbone_type == 'vgg16_bn':
            self.backbone = OrderedDict()
            in_channels = 3
            for v in self.vgg_cfg['D']:
                if v == 'M':
                    self.backbone['mp{}'.format(
                        self._add_mp_count())] = self.maxpool
                else:
                    self.backbone['conv{}'.format(self._add_conv_count())] = (
                        M.conv2d_block(
                            in_channels=in_channels,
                            out_channels=v,
                            kernel_size=3,
                            padding=1,
                            init_type=self.init_type,
                            activation=self.activation,
                            use_batchnorm=self.bn
                        ))
                    in_channels = v
        # print(self.backbone)
        self.feature_extract = (
            M.conv2d_block(
                in_channels=in_channels,
                out_channels=256,
                kernel_size=3,
                padding=1,
                init_type=self.init_type,
                activation=self.activation,
                use_batchnorm=self.bn
            ))

        self.reduction_a = ReductionModuleA(in_channels=256, out_channels=256)
        self.reduction_b = ReductionModuleB(in_channels=128, out_channels=128)

    def _add_conv_count(self):
        self.conv_count += 1
        return self.conv_count

    def _add_mp_count(self):
        self.mp_count += 1
        return self.mp_count

    def forward(self, x):
        skip_connect = []
        for k, layer in self.backbone.items():
            x = layer(x)
            if 'mp' in k and k != 'mp5':
                skip_connect.append(x)
        x = self.feature_extract(x)
        print('feature_extract', x.shape)
        x = self.reduction_a(x)
        print('reduction_a', x.shape)
        return x


if __name__ == "__main__":
    GiN = GradientInferenceNetwork()
    IiN = ImageInferenceNetwork(backbone_type='vgg16_bn')
    inputs = torch.randn(4, 3, 224, 288)
    if torch.cuda.is_available():
        inputs = inputs.cuda()
        IiN = IiN.cuda()
    output = IiN(inputs)
    print(output.shape)
