import sys
import torch
import torch.nn as nn
sys.path.append('../..')
import base_model.nn_module as M


class LayerConv(nn.Module):
    def __init__(self, in_channels, out_channels, layers=3, init_type="xavier", activation=nn.ReLU(), norm_type=None):
        super(LayerConv, self).__init__()
        convs = []
        for i in range(layers):
            if i == 0:
                convs.append(
                    M.conv2d_block(
                        in_channels=in_channels,
                        out_channels=out_channels,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                        init_type=init_type,
                        activation=activation,
                        norm_type=norm_type
                        )
                    )
            else:
                convs.append(
                    M.conv2d_block(
                        in_channels=out_channels,
                        out_channels=out_channels,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                        init_type=init_type,
                        activation=activation,
                        norm_type=norm_type
                        )
                    )
        self.main = nn.Sequential(*convs)

    def forward(self, x):
        x = self.main(x)
        return x


class DownsampleConv(nn.Module):
    def __init__(self, in_channels, out_channels, layers, downsample_type="maxpool", init_type="xavier", activation=nn.ReLU(), norm_type='BN'):
        super(DownsampleConv, self).__init__()
        if downsample_type == "maxpool":
            self.downsample = nn.MaxPool2d(kernel_size=2)
        elif downsample_type == "conv_stride2":
            self.downsample = M.conv2d_block(
                    in_channels=in_channels,
                    out_channels=in_channels,
                    kernel_size=4,
                    stride=2,
                    padding=1,
                    init_type=init_type,
                    activation=activation,
                    norm_type=norm_type
                    )
        else:
            raise ValueError("invalid downsample type: {}".format(downsample_type))
        convs = []
        for i in range(layers):
            if i == 0:
                convs.append(
                    M.conv2d_block(
                        in_channels=in_channels,
                        out_channels=out_channels,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                        init_type=init_type,
                        activation=activation,
                        norm_type=norm_type
                        )
                    )
            else:
                convs.append(
                    M.conv2d_block(
                        in_channels=out_channels,
                        out_channels=out_channels,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                        init_type=init_type,
                        activation=activation,
                        norm_type=norm_type
                        )
                    )
        self.main = nn.Sequential(*convs)

    def forward(self, x):
        x = self.downsample(x)
        x = self.main(x)
        return x


class UpsampleConv(nn.Module):
    def __init__(self, in_channels, out_channels, layers, upsample_type="nearest", init_type="xavier", activation=nn.ReLU(), norm_type='BN'):
        super(UpsampleConv, self).__init__()
        convs = []
        for i in range(layers):
            if i == 0:
                convs.append(
                    M.conv2d_block(
                        in_channels=in_channels,
                        out_channels=out_channels,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                        init_type=init_type,
                        activation=activation,
                        norm_type=norm_type
                        )
                    )
            else:
                convs.append(
                    M.conv2d_block(
                        in_channels=out_channels,
                        out_channels=out_channels,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                        init_type=init_type,
                        activation=activation,
                        norm_type=norm_type
                        )
                    )
        self.main = nn.Sequential(*convs)
        if upsample_type == "nearest":
            self.upsample = lambda x: nn.functional.interpolate(x, scale_factor=2, mode='nearest')
        elif upsample_type == "conv_transpose_stride2":
            self.upsample = M.deconv2d_block(
                    in_channels=out_channels,
                    out_channels=out_channels,
                    kernel_size=4,
                    stride=2,
                    padding=1,
                    init_type=init_type,
                    activation=activation,
                    norm_type=norm_type
                    )
        else:
            raise ValueError("invalid upsample type: {}".format(upsample_type))

    def forward(self, x):
        x = self.main(x)
        x = self.upsample(x)
        return x


class MidConv(nn.Module):
    def __init__(self, in_channels, out_channels, layers, downsample_type="maxpool",
                 upsample_type="nearest", init_type="xavier", activation=nn.ReLU(), norm_type='BN'):
        super(MidConv, self).__init__()
        if downsample_type == "maxpool":
            self.downsample = nn.MaxPool2d(kernel_size=2)
        elif downsample_type == "conv_stride2":
            self.downsample = M.conv2d_block(
                    in_channels=in_channels,
                    out_channels=in_channels,
                    kernel_size=4,
                    stride=2,
                    padding=1,
                    init_type=init_type,
                    activation=activation,
                    norm_type=norm_type
                    )
        else:
            raise ValueError("invalid downsample type: {}".format(downsample_type))
        convs = []
        for i in range(layers):
            if i == 0:
                convs.append(
                    M.conv2d_block(
                        in_channels=in_channels,
                        out_channels=out_channels,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                        init_type=init_type,
                        activation=activation,
                        norm_type=norm_type
                        )
                    )
            else:
                convs.append(
                    M.conv2d_block(
                        in_channels=out_channels,
                        out_channels=out_channels,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                        init_type=init_type,
                        activation=activation,
                        norm_type=norm_type
                        )
                    )
        self.main = nn.Sequential(*convs)

        if upsample_type == "nearest":
            self.upsample = lambda x: nn.functional.interpolate(x, scale_factor=2, mode='nearest')
        elif upsample_type == "conv_transpose_stride2":
            self.upsample = M.deconv2d_block(
                    in_channels=out_channels,
                    out_channels=out_channels,
                    kernel_size=4,
                    stride=2,
                    padding=1,
                    init_type=init_type,
                    activation=activation,
                    norm_type=norm_type
                    )
        else:
            raise ValueError("invalid upsample type: {}".format(upsample_type))

    def forward(self, x):
        x = self.downsample(x)
        x = self.main(x)
        x = self.upsample(x)
        return x


class BurstDeblurMP(nn.Module):
    #encoder_list = [32, 64, 128, 256]
    #decoder_list = [256, 128, 64, 32]
    encoder_list = [16, 24, 48, 64]
    decoder_list = [64, 48, 24, 16]

    def __init__(self, in_channels=1, layers=2, init_type="xavier", norm_type='WN'):
        super(BurstDeblurMP, self).__init__()
        self.init_type = init_type
        self.bn = norm_type
        self.activation = nn.ReLU(inplace=True)
        self.layers = layers

        self.input_conv = LayerConv(
                    in_channels=in_channels,
                    out_channels=self.encoder_list[0],
                    layers=self.layers,
                    init_type=self.init_type,
                    activation=self.activation,
                    norm_type=self.bn)
        self.fusion1 = M.conv2d_block(
                    in_channels=2*self.encoder_list[0],
                    out_channels=self.encoder_list[0],
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    init_type=self.init_type,
                    activation=self.activation,
                    norm_type=self.bn
                )
        self.downsample_conv1 = DownsampleConv(
                    in_channels=self.encoder_list[0],
                    out_channels=self.encoder_list[1],
                    layers=self.layers,
                    init_type=self.init_type,
                    activation=self.activation,
                    norm_type=self.bn
                )
        self.fusion2 = M.conv2d_block(
                    in_channels=2*self.encoder_list[1],
                    out_channels=self.encoder_list[1],
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    init_type=self.init_type,
                    activation=self.activation,
                    norm_type=self.bn
                )
        self.downsample_conv2 = DownsampleConv(
                    in_channels=self.encoder_list[1],
                    out_channels=self.encoder_list[2],
                    layers=self.layers,
                    init_type=self.init_type,
                    activation=self.activation,
                    norm_type=self.bn
                )
        self.fusion3 = M.conv2d_block(
                    in_channels=2*self.encoder_list[2],
                    out_channels=self.encoder_list[2],
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    init_type=self.init_type,
                    activation=self.activation,
                    norm_type=self.bn
                )
        self.mid = MidConv(
                    in_channels=self.encoder_list[2],
                    out_channels=self.decoder_list[0],
                    layers=self.layers,
                    init_type=self.init_type,
                    activation=self.activation,
                    norm_type=self.bn
                )
        self.fusion4 = M.conv2d_block(
                    in_channels=2*self.decoder_list[0],
                    out_channels=self.decoder_list[0],
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    init_type=self.init_type,
                    activation=self.activation,
                    norm_type=self.bn
                )
        self.upsample_conv1 = UpsampleConv(
                    in_channels=self.decoder_list[0] + self.encoder_list[2],
                    out_channels=self.decoder_list[1],
                    layers=self.layers,
                    init_type=self.init_type,
                    activation=self.activation,
                    norm_type=self.bn
                )
        self.fusion5 = M.conv2d_block(
                    in_channels=2*self.decoder_list[1],
                    out_channels=self.decoder_list[1],
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    init_type=self.init_type,
                    activation=self.activation,
                    norm_type=self.bn
                )
        self.upsample_conv2 = UpsampleConv(
                    in_channels=self.decoder_list[1] + self.encoder_list[1],
                    out_channels=self.decoder_list[2],
                    layers=self.layers,
                    init_type=self.init_type,
                    activation=self.activation,
                    norm_type=self.bn
                )
        self.fusion6 = M.conv2d_block(
                    in_channels=2*self.decoder_list[2],
                    out_channels=self.decoder_list[2],
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    init_type=self.init_type,
                    activation=self.activation,
                    norm_type=self.bn
                )
        self.output_conv1 = LayerConv(
                    in_channels=self.decoder_list[2] + self.encoder_list[0],
                    out_channels=self.decoder_list[3],
                    layers=self.layers,
                    init_type=self.init_type,
                    activation=self.activation,
                    norm_type=self.bn
                )
        self.output_conv2 = LayerConv(
                    in_channels=self.decoder_list[3],
                    out_channels=in_channels,
                    layers=1,
                    init_type=self.init_type,
                    activation=None,
                    norm_type=None
                )

    def max_op(self, layer, x, num=6, skip_connect=None):
        _, c, h, w = x.shape
        x_max = x.view(-1, num, c, h, w).max(dim=1)[0]
        x_r = []
        for i in range(num):
            if skip_connect is None:
                x_t = torch.cat([x_max, x[i::num]], dim=1)
            else:
                x_t = torch.cat([x_max, x[i::num], skip_connect[i::num]], dim=1)
            x_r.append(layer(x_t))
        _, c, h, w = x_r[0].shape
        return torch.stack(x_r, dim=1).view(-1, c, h, w)


    def forward(self, x):
        b, n, c, h, w = x.shape
        x = x.view(-1, c, h, w)

        x1 = self.input_conv(x)
        x1 = self.max_op(self.fusion1, x1)
        x2 = self.downsample_conv1(x1)
        x2 = self.max_op(self.fusion2, x2)
        x3 = self.downsample_conv2(x2)
        x3 = self.max_op(self.fusion3, x3)
        x4 = self.mid(x3)
        x4 = self.max_op(self.fusion4, x4)
        x5 = self.upsample_conv1(torch.cat([x4, x3], dim=1))
        x5 = self.max_op(self.fusion5, x5)
        x6 = self.upsample_conv2(torch.cat([x5, x2], dim=1))
        x6 = self.max_op(self.fusion6, x6)
        x7 = self.output_conv1(torch.cat([x6, x1], dim=1))

        _, c, h, w = x7.shape
        x7 = x7.view(b, n, c, h, w).max(dim=1)[0]
        x8 = self.output_conv2(x7)
        return x8

    def forward_old(self, x):
        b, n, c, h, w = x.shape
        x = x.view(-1, c, h, w)
        x1 = self.input_conv(x)

        _, c, h, w = x1.shape
        x1_d = x1.view(b, n, c, h, w)
        x1_g = torch.max(x1_d, dim=1)[0]
        x1_g = x1_g.unsqueeze(1).repeat(1, n, 1, 1, 1).view(-1, c, h, w)
        x1_c = torch.cat([x1, x1_g], dim=1)
        x1_c = self.fusion1(x1_c)

        x2 = self.downsample_conv1(x1_c)

        _, c, h, w = x2.shape
        x2_d = x2.view(b, n, c, h, w)
        x2_g = torch.max(x2_d, dim=1)[0]
        x2_g = x2_g.unsqueeze(1).repeat(1, n, 1, 1, 1).view(-1, c, h, w)
        x2_c = torch.cat([x2, x2_g], dim=1)
        x2_f = self.fusion2(x2_c)

        x3 = self.downsample_conv2(x2_f)

        _, c, h, w = x3.shape
        x3_d = x3.view(b, n, c, h, w)
        x3_g = torch.max(x3_d, dim=1)[0]
        x3_g = x3_g.unsqueeze(1).repeat(1, n, 1, 1, 1).view(-1, c, h, w)
        x3_c = torch.cat([x3, x3_g], dim=1)
        x3_f = self.fusion3(x3_c)

        x4 = self.mid(x3_f)

        _, c, h, w = x4.shape
        x4_d = x4.view(b, n, c, h, w)
        x4_g = torch.max(x4_d, dim=1)[0]
        x4_g = x4_g.unsqueeze(1).repeat(1, n, 1, 1, 1).view(-1, c, h, w)
        x4_c = torch.cat([x4, x4_g, x3], dim=1)
        x4_f = self.fusion4(x4_c)

        x5 = self.upsample_conv1(x4_f)

        _, c, h, w = x5.shape
        x5_d = x5.view(b, n, c, h, w)
        x5_g = torch.max(x5_d, dim=1)[0]
        x5_g = x5_g.unsqueeze(1).repeat(1, n, 1, 1, 1).view(-1, c, h, w)
        x5_c = torch.cat([x5, x5_g, x2], dim=1)
        x5_f = self.fusion5(x5_c)

        x6 = self.upsample_conv2(x5_f)

        _, c, h, w = x6.shape
        x6_d = x6.view(b, n, c, h, w)
        x6_g = torch.max(x6_d, dim=1)[0]
        x6_g = x6_g.unsqueeze(1).repeat(1, n, 1, 1, 1).view(-1, c, h, w)
        x6_c = torch.cat([x6, x6_g, x1], dim=1)
        x6_f = self.fusion6(x6_c)

        x7 = self.output_conv1(x6_f)

        _, c, h, w = x7.shape
        x7_d = x7.view(b, n, c, h, w)
        x = torch.max(x7_d, dim=1)[0]
        x8 = self.output_conv2(x)
        return x8


def unit_test_burst_deblur_mp():
    net = BurstDeblurMP(in_channels=3, layers=2, norm_type='WN')
    inputs = torch.randn(8, 6, 3, 256, 256)
    if torch.cuda.is_available():
        net = net.cuda()
        inputs = inputs.cuda()
    output = net.forward_new(inputs)
    print(torch.cuda.max_memory_allocated())


if __name__ == "__main__":
    unit_test_burst_deblur_mp()
    '''
    x1 = torch.randn(4, 8, 3)
    print(x1)
    x2 = torch.max(x1, dim=0)[0]
    print(x2)
    '''

