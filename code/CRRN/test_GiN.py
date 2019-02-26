import pytest
import torch.nn as nn
from numpy.testing import assert_allclose
from receptivefield.pytorch import PytorchReceptiveField
from receptivefield.image import get_default_image
from receptivefield.types import ImageShape
from collections import OrderedDict
import torch
import sys

sys.path.append('../')
import base_model.nn_module as M


class Linear(nn.Module):
    """An identity activation function"""

    def forward(self, x):
        return x


class GradientInferenceNetwork(nn.Module):
    encoder_list = [64, 128, 256, 512, 512]
    decoder_list = [256, 128, 64, 32]

    def __init__(self, init_type="xavier", use_batchnorm=False, use_maxpool=False, DEBUG=False):
        super(GradientInferenceNetwork, self).__init__()
        self.init_type = init_type
        self.bn = use_batchnorm
        self.use_maxpool = use_maxpool
        #self.activation = nn.ReLU(inplace=True)
        self.activation = Linear()##change the activation

        self.feature_maps = None

        self.encoder = OrderedDict()
        in_channels = 4
        for i in range(len(self.encoder_list)):
            self.encoder['conv{}'.format(i + 1)] = M.conv2d_block(
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
                self.encoder['conv_next{}'.format(i + 1)] = M.conv2d_block(
                    in_channels=in_channels,
                    out_channels=self.encoder_list[i],
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    init_type=self.init_type,
                    activation=self.activation,
                    use_batchnorm=self.bn
                )
                self.encoder['downsample{}'.format(i + 1)] = nn.MaxPool2d(kernel_size=2)
            else:
                self.encoder['conv_downsample{}'.format(i + 1)] = M.conv2d_block(
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
            self.decoder['conv{}'.format(i + 1)] = M.conv2d_block(
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
            self.decoder['deconv{}'.format(i + 1)] = M.deconv2d_block(
                in_channels=in_channels,
                out_channels=self.decoder_list[i],
                kernel_size=4,
                stride=2,
                padding=1,
                init_type=self.init_type,
                activation=self.activation,
                use_batchnorm=self.bn
            )
            in_channels += self.encoder_list[3 - i]  # concat

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
        self.feature_maps = []
        skip_connect = []
        for k, layer in self.encoder.items():
            x = layer(x)
            if 'downsample' in k and 'downsample5' not in k:
                skip_connect.append(x)
            self.feature_maps.append(x)  #add feature_maps

        x = self.mid(x)
        self.feature_maps.append(x)
        '''
        gradient_guide = []
        count = 1
        for k, layer in self.decoder.items():
            x = layer(x)
            if 'deconv' in k:
                gradient_guide.append(x)
                x = torch.cat([x, skip_connect[-count]], dim=1)
                count += 1
            self.feature_maps.append(x)  #add feature_maps
        estimate_gradient_B = self.end(x)
        return estimate_gradient_B, gradient_guide
        '''

def model_fn() -> nn.Module:
    model = GradientInferenceNetwork()
    model.eval()
    return model


def test_example_network():
    input_shape = [224, 288, 4]
    rf = PytorchReceptiveField(model_fn)
    rf_params = rf.compute(input_shape=ImageShape(*input_shape))

    '''
    assert_allclose(rf_params[0].rf.size, (6, 6))
    assert_allclose(rf_params[0].rf.stride, (2, 2))

    rs = 6 + (2 + 2 + 1) * 2
    assert_allclose(rf_params[1].rf.size, (rs, rs))
    assert_allclose(rf_params[1].rf.stride, (4, 4))

    rs = 6 + (2 + 2 + 1) * 2 + (2 + 2 + 1) * 4
    assert_allclose(rf_params[2].rf.size, (rs, rs))
    assert_allclose(rf_params[2].rf.stride, (8, 8))
    '''
    # rf.plot_gradient_at(fm_id=1, point=(9, 9))
    # rf.plot_rf_grids(get_default_image(input_shape, name='cat'),plot_naive_rf=True, figsize=(20, 12))


if __name__ == "__main__":
    pytest.main([__file__])
    test_example_network()
