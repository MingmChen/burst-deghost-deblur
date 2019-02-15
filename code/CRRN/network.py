import sys
from collections import OrderedDict
import torch
import torch.nn as nn
sys.path.append('..')
import base_model.nn_module as M


class GradientInferenceNetwork(nn.Module):
    encoder_list = [64, 128, 256, 512, 512]
    decoder_list = [256, 128, 64, 32]

    def __init__(self, init_type="xavier", use_batchnorm=True, use_maxpool=False, DEBUG=False):
        super(GradientInferenceNetwork, self).__init__()
        self.init_type = init_type
        self.bn = use_batchnorm
        self.use_maxpool = use_maxpool
        self.activation = nn.ReLU(inplace=True)

        self.encoder = OrderedDict()
        in_channels = 4
        for i in range(len(self.encoder_list)):
            self.encoder['conv{}'.format(i+1)] = M.conv2d_block(
                        in_channels=in_channels,
                        out_channels=self.encoder_list[i],
                        kernel_size=3,
                        stride=1,
                        padding=1,
                        init_type=self.init_type,
                        activation=self.activation,
                        use_batchnorm=self.bn
                    )
            in_channels = self.encoder_list[i]
            if use_maxpool:
                self.encoder['conv_next{}'.format(i+1)] = M.conv2d_block(
                            in_channels=in_channels,
                            out_channels=self.encoder_list[i],
                            kernel_size=3,
                            stride=1,
                            padding=1,
                            init_type=self.init_type,
                            activation=self.activation,
                            use_batchnorm=self.bn
                        )
                self.encoder['downsample{}'.format(i+1)] = nn.MaxPool2d(kernel_size=2)
            else:
                self.encoder['conv_downsample{}'.format(i+1)] = M.conv2d_block(
                            in_channels=in_channels,
                            out_channels=self.encoder_list[i],
                            kernel_size=4,
                            stride=2,
                            padding=1,
                            init_type=self.init_type,
                            activation=self.activation,
                            use_batchnorm=self.bn
                        )

        self.mid = nn.Sequential(
                M.conv2d_block(
                    in_channels=in_channels,
                    out_channels=1024,
                    kernel_size=7,
                    stride=1,
                    padding=3,
                    init_type=self.init_type,
                    activation=self.activation,
                    use_batchnorm=self.bn
                    ),
                M.conv2d_block(
                    in_channels=1024,
                    out_channels=512,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    init_type=self.init_type,
                    activation=self.activation,
                    use_batchnorm=self.bn
                    )
                )

        in_channels = 512
        self.decoder = OrderedDict()
        for i in range(len(self.decoder_list)):
            self.decoder['conv{}'.format(i+1)] = M.conv2d_block(
                        in_channels=in_channels,
                        out_channels=self.decoder_list[i],
                        kernel_size=3,
                        stride=1,
                        padding=1,
                        init_type=self.init_type,
                        activation=self.activation,
                        use_batchnorm=self.bn
                    )
            in_channels = self.decoder_list[i]
            self.decoder['deconv{}'.format(i+1)] = M.deconv2d_block(
                        in_channels=in_channels,
                        out_channels=self.decoder_list[i],
                        kernel_size=4,
                        stride=2,
                        padding=1,
                        init_type=self.init_type,
                        activation=self.activation,
                        use_batchnorm=self.bn
                    )
            in_channels += self.encoder_list[3-i]  # concat

        self.end = nn.Sequential(
                M.conv2d_block(
                    in_channels=in_channels,
                    out_channels=64,
                    kernel_size=4,
                    stride=2,
                    padding=1,
                    init_type=self.init_type,
                    activation=self.activation,
                    use_batchnorm=self.bn
                    ),
                M.conv2d_block(
                    in_channels=64,
                    out_channels=1,
                    kernel_size=5,
                    stride=1,
                    padding=2,
                    init_type=self.init_type,
                    activation=None,
                    use_batchnorm=None
                    )
                )
        if DEBUG:
            print(self.encoder)
            print(self.mid)
            print(self.decoder)
            print(self.end)

    def to_cuda(self):
        for k, layer in self.encoder.items():
            layer = layer.cuda()
        for k, layer in self.decoder.items():
            layer = layer.cuda()

    def forward(self, x):
        skip_connect = []
        for k, layer in self.encoder.items():
            x = layer(x)
            if 'downsample' in k and 'downsample5' not in k:
                skip_connect.append(x)

        x = self.mid(x)

        gradient_guide = []
        count = 1
        for k, layer in self.decoder.items():
            x = layer(x)
            if 'deconv' in k:
                gradient_guide.append(x)
                x = torch.cat([x, skip_connect[-count]], dim=1)
                count += 1

        estimate_gradient_B = self.end(x)
        return estimate_gradient_B, gradient_guide


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

    def forward(x):
        x = self.feature_extract(x)

        x = self.reduction_a(x)

        x = self.inception1(x)
        x = torch.cat([x, skip_connect[-1], gradient_guide[0]], dim=1)
        x = self.inception2(x)
        x = torch.cat([x, skip_connect[-2], gradient_guide[1]], dim=1)

        x = self.reduction_b(x)

        x = self.inception3(x)
        x = torch.cat([x, skip_connect[-3], gradient_guide[2]], dim=1)
        x = self.inception4(x)
        x = torch.cat([x, skip_connect[-4], gradient_guide[3]], dim=1)
        x = self.inception5(x)

        x = self.conv_transition(x)
        estimate_B = self.conv_estimate_B(x)
        estimate_R = origin_input - estimate_B
        return estimate_B, estimate_R


def unit_test_IiN(gradient_guide):
    IiN = ImageInferenceNetwork(backbone_type='vgg16_bn')
    inputs = torch.randn(4, 3, 224, 288)
    if torch.cuda.is_available():
        inputs = inputs.cuda()
        IiN = IiN.cuda()
        IiN.to_cuda()
        for item in gradient_guide:
            item = item.cuda()
    estimate_B, estimate_R = IiN(inputs, gradient_guide)
    print('estimate_B', estimate_B.shape)
    print('estimate_R', estimate_R.shape)


def unit_test_GiN():
    GiN = GradientInferenceNetwork()
    inputs = torch.randn(4, 4, 224, 288)
    if torch.cuda.is_available():
        inputs = inputs.cuda()
        GiN = GiN.cuda()
        GiN.to_cuda()
    estimate_gradient_B, gradient_guide = GiN(inputs)
    print('estimate_gradient_B', estimate_gradient_B.shape)
    print('number of gradient_guide', len(gradient_guide))
    for i in range(len(gradient_guide)):
        print('gradient_guide{}'.format(i+1), gradient_guide[i].shape)
    return gradient_guide


if __name__ == "__main__":
    gradient_guide = unit_test_GiN()
    #unit_test_IiN(gradient_guide)
