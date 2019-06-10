# -*- coding:utf-8 -*-
'''
version 2:  roughly encapsulation
@butub
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
sys.path.append('../')
from utils.log import time_log
import base_model.nn_module as M
dtype = torch.cuda.FloatTensor
dtype_long = torch.cuda.LongTensor


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
                        pad_type='reflect',
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
                        pad_type='reflect',
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
    def __init__(self, in_channels, out_channels, layers, downsample_type="maxpool", init_type="xavier", activation=nn.ReLU(), norm_type='bn'):
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
                pad_type='reflect',
                init_type=init_type,
                activation=activation,
                norm_type=norm_type
            )
        else:
            raise ValueError(
                "invalid downsample type: {}".format(downsample_type))
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
                        pad_type='reflect',
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
                        pad_type='reflect',
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


class MidConv(nn.Module):
    def __init__(self, in_channels, out_channels, layers, downsample_type="maxpool",
                 upsample_type="bilinear", init_type="xavier", activation=nn.ReLU(), norm_type='bn'):
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
                pad_type='reflect',
                init_type=init_type,
                activation=activation,
                norm_type=norm_type
            )
        else:
            raise ValueError(
                "invalid downsample type: {}".format(downsample_type))
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
                        pad_type='reflect',
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
                        pad_type='reflect',
                        init_type=init_type,
                        activation=activation,
                        norm_type=norm_type
                    )
                )
        self.main = nn.Sequential(*convs)

        if upsample_type == "bilinear":
            self.upsample = lambda x: F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        elif upsample_type == "conv_transpose_stride2":
            self.upsample = M.deconv2d_block(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=4,
                stride=2,
                padding=1,
                pad_type='reflect',
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


class UpsampleConv(nn.Module):
    def __init__(self, in_channels, out_channels, layers=2, upsample_type="bilinear", init_type="xavier", activation=nn.ReLU(), norm_type='bn'):
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
                        pad_type='reflect',
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
                        pad_type='reflect',
                        init_type=init_type,
                        activation=activation,
                        norm_type=norm_type
                    )
                )
        self.main = nn.Sequential(*convs)
        if upsample_type == "bilinear":
            self.upsample = lambda x: F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        elif upsample_type == "conv_transpose_stride2":
            self.upsample = M.deconv2d_block(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=4,
                stride=2,
                padding=1,
                pad_type='reflect',
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


class SynthesizeBlur(nn.Module):
    encoder_list = [32, 64, 128, 256]
    decoder_list = [256, 128, 64, 32]

    def __init__(self, in_channels=6, layers=3, init_type='xavier', norm_type=None):
        super(SynthesizeBlur, self).__init__()
        if torch.cuda.is_available():
            self.device = 'cuda'
        else:
            self.device = 'cpu'
        self.layers = layers
        self.init_type = init_type
        self.norm_type = norm_type
        self.LeakRate = 0.2
        self.activation = nn.LeakyReLU(self.LeakRate)
        self.N = 17  # N evnely-spaced discrete samples

        #
        self.input_conv = LayerConv(
            in_channels=in_channels,
            out_channels=self.encoder_list[0],
            layers=self.layers,
            activation=self.activation,
            norm_type=self.norm_type)

        self.downsample1 = DownsampleConv(
            in_channels=self.encoder_list[0],
            out_channels=self.encoder_list[1],
            layers=self.layers,
            downsample_type="maxpool",
            init_type=self.init_type,
            activation=self.activation,
            norm_type=self.norm_type
        )
        self.downsample2 = DownsampleConv(
            in_channels=self.encoder_list[1],
            out_channels=self.encoder_list[2],
            layers=self.layers,
            downsample_type="maxpool",
            init_type=self.init_type,
            activation=self.activation,
            norm_type=self.norm_type
        )

        self.downsample3 = DownsampleConv(
            in_channels=self.encoder_list[2],
            out_channels=self.encoder_list[3],
            layers=self.layers,
            downsample_type="maxpool",
            init_type=self.init_type,
            activation=self.activation,
            norm_type=self.norm_type
        )

        self.mid = MidConv(
            in_channels=self.encoder_list[3],
            out_channels=self.decoder_list[0],
            layers=self.layers,
            downsample_type='maxpool',
            upsample_type='bilinear',
            init_type=self.init_type,
            activation=self.activation,
            norm_type=self.norm_type)
        self.bottom1 = M.conv2d_block(
            in_channels=self.decoder_list[0],
            out_channels=self.decoder_list[0],
            kernel_size=3,
            stride=1,
            padding=1,
            pad_type='reflect',
            init_type=self.init_type,
            activation=self.activation,
            norm_type=self.norm_type
        )

        self.upsample_conv1 = UpsampleConv(
            in_channels=self.decoder_list[0]+self.encoder_list[3],
            out_channels=self.decoder_list[0],
            layers=2,
            upsample_type='bilinear',
            init_type=self.init_type,
            activation=self.activation,
            norm_type=self.norm_type
        )
        self.bottom2 = M.conv2d_block(
            in_channels=self.decoder_list[0],
            out_channels=self.decoder_list[1],
            kernel_size=3,
            stride=1,
            padding=1,
            pad_type='reflect',
            init_type=self.init_type,
            activation=self.activation,
            norm_type=self.norm_type
        )


        self.upsample_conv2 = UpsampleConv(
            in_channels=self.decoder_list[1] + self.encoder_list[2],
            out_channels=self.decoder_list[1],
            layers=self.layers,
            upsample_type='bilinear',
            init_type=self.init_type,
            activation=self.activation,
            norm_type=self.norm_type

        )
        self.bottom3 = M.conv2d_block(
            in_channels=self.decoder_list[1],
            out_channels=self.decoder_list[2],
            kernel_size=3,
            stride=1,
            padding=1,
            pad_type='reflect',
            init_type=self.init_type,
            activation=self.activation,
            norm_type=self.norm_type
        )

        self.upsample_conv3 = UpsampleConv(
            in_channels=self.decoder_list[2] + self.encoder_list[1],
            out_channels=self.decoder_list[2],
            layers=self.layers,
            upsample_type='bilinear',
            init_type=self.init_type,
            activation=self.activation,
            norm_type=self.norm_type
        )
        self.bottom4 = M.conv2d_block(
            in_channels=self.decoder_list[2],
            out_channels=self.decoder_list[3],
            kernel_size=3,
            stride=1,
            padding=1,
            pad_type='reflect',
            init_type=self.init_type,
            activation=self.activation,
            norm_type=self.norm_type
        )

        self.output_conv = LayerConv(
            in_channels=self.decoder_list[3] + self.encoder_list[0],
            out_channels=self.decoder_list[3],
            layers=self.layers,
            init_type='xavier',
            activation=self.activation,
            norm_type=self.norm_type
        )

        '''
        outputs:  feeds into line prediction layer
        '''
        self.offset1 = nn.Conv2d(32, 2, kernel_size=1,
                                 stride=1, padding=0)
        self.offset2 = nn.Conv2d(32, 2, kernel_size=1,
                                 stride=1, padding=0)
        self.weight1 = nn.Conv2d(
            32, self.N, kernel_size=1, stride=1, padding=0)
        self.weight2 = nn.Conv2d(
            32, self.N, kernel_size=1, stride=1, padding=0)


    @time_log
    def forward(self, inp1, inp2):
        '''
        input: NCHW, C:6, H:256, W:256
        output: synthesize image
        '''

        self.shape = inp1.shape
        self.minibatch = inp1.shape[0]

        x = torch.cat([inp1, inp2], dim=1)

        x_conv = self.input_conv(x)
        x1 = self.downsample1(x_conv)
        x2 = self.downsample2(x1)
        x3 = self.downsample3(x2)

        x4 = self.mid(x3)
        x4_b = self.bottom1(x4)
        x4_c = torch.cat([x4_b, x3], dim=1)

        x5 = self.upsample_conv1(x4_c)
        x5_b = self.bottom2(x5)
        x5_c = torch.cat([x5_b, x2], dim=1)

        x6 = self.upsample_conv2(x5_c)
        x6_b = self.bottom3(x6)
        x6_c = torch.cat([x6_b, x1], dim=1)

        x7 = self.upsample_conv3(x6_c)
        x7_b = self.bottom4(x7)
        x7_c = torch.cat([x7_b, x_conv], dim=1)

        x8 = self.output_conv(x7_c)

        offset1 = self.offset1(x8)   # Nx2xHxW
        offset2 = self.offset2(x8)   # Nx2xHxW
        weight1 = self.weight1(x8)   # Nx17xHxW
        weight2 = self.weight2(x8)   # Nx17xHxW

        sample = self.line_prediction(
            offset1, offset2, weight1, weight2, inp1, inp2)

        return sample

    @time_log
    def violent_cycle(self, _offset1, _offset2, _weight1, _weight2, _inp1, _inp2):
        B, C, H, W = _inp1.shape
        norm_factor = torch.FloatTensor([H/2, W/2]).to(self.device)
        theta = torch.tensor([[[1, 0, 0], [0, 1, 0]]], dtype=torch.float).repeat(
            self.minibatch, 1, 1).to(self.device)  # Nx2x3
        grid = nn.functional.affine_grid(theta, self.shape).mul_(norm_factor)

        sample = torch.zeros_like(_inp1).to(self.device)
        for n in range(self.N):
            # Nx2xHxW --> NxHxWx2
            grid1 = torch.clamp(grid + (n / self.N) *
                                (_offset1.permute(0, 2, 3, 1)), -1, 1)
            grid1.div_(norm_factor)
            sample_n1 = nn.functional.grid_sample(
                _inp1, grid1, padding_mode='border')
            sample += (_weight1[:, n, :, :]).unsqueeze(1) * sample_n1

        for n in range(self.N):
            grid2 = torch.clamp(grid + (n / self.N) *
                                (_offset2.permute(0, 2, 3, 1)), -1, 1)
            grid2.div_(norm_factor)
            sample_n2 = nn.functional.grid_sample(
                _inp2, grid2, padding_mode='border')
            sample += (_weight2[:, n, :, :]).unsqueeze(1) * sample_n2

        return sample

    @time_log
    def line_prediction(self, offset1, offset2, weight1, weight2, img1, img2):
        N = self.N
        B, C, H, W = img1.shape
        norm_factor = torch.FloatTensor([H/2, W/2]).to(self.device)
        theta = torch.FloatTensor([1, 0, 0, 0, 1, 0]).view(
            1, 2, 3).repeat(B, 1, 1).to(self.device)
        grid = F.affine_grid(theta, self.shape)
        grid.mul_(norm_factor).unsqueeze_(0)

        offset1_sample = offset1.permute(
            0, 2, 3, 1).contiguous().unsqueeze(0).repeat(N, 1, 1, 1, 1)
        offset2_sample = offset2.permute(
            0, 2, 3, 1).contiguous().unsqueeze(0).repeat(N, 1, 1, 1, 1)
        sample_factor = torch.FloatTensor(
            [x for x in range(N)]).view(N, 1, 1, 1, 1).to(self.device)

        offset1_sample.mul_(sample_factor).add_(grid).div_(norm_factor)
        offset2_sample.mul_(sample_factor).add_(grid).div_(norm_factor)

        sample1 = []
        sample2 = []
        for n in range(N):
            sample1.append(F.grid_sample(
                img1, offset1_sample[n], padding_mode='border'))
            sample2.append(F.grid_sample(
                img2, offset2_sample[n], padding_mode='border'))

        sample1 = torch.stack(sample1, dim=1)
        sample2 = torch.stack(sample2, dim=1)

        weight1 = weight1.view(B, N, 1, H, W)
        weight2 = weight2.view(B, N, 1, H, W)

        sample = (weight1*sample1 + weight2*sample2).sum(dim=1)

        return sample

class MotionBlurArgumentLayer(nn.Module):
    def __init__(self, in_channels=32, weight_num=None, init_type="xavier"):
        super(MotionBlurArgumentLayer, self).__init__()
        self.offset1 = M.conv2d_block(in_channels, 2, 1, 1, 0, init_type=init_type, activation=None, norm_type=None)
        self.offset2 = M.conv2d_block(in_channels, 2, 1, 1, 0, init_type=init_type, activation=None, norm_type=None)
        self.weight1 = M.conv2d_block(in_channels, weight_num, 1, 1, 0, init_type=init_type, activation=None, norm_type=None)
        self.weight2 = M.conv2d_block(in_channels, weight_num, 1, 1, 0, init_type=init_type, activation=None, norm_type=None)

    def forward(self, x):
        o1 = self.offset1(x)
        o2 = self.offset2(x)
        w1 = self.weight1(x)
        w2 = self.weight2(x)
        return o1, o2, w1, w2


class SynthesizeBlurBranch(nn.Module):
    encoder_list = [32, 64, 128, 256]
    decoder_list = [256, 128, 64, 32]

    def __init__(self, in_channels=6, layers=3, branch_num=[9, 17, 33], init_type='xavier', norm_type=None):
        super(SynthesizeBlurBranch, self).__init__()
        if torch.cuda.is_available():
            self.device = 'cuda'
        else:
            self.device = 'cpu'
        self.layers = layers
        self.init_type = init_type
        self.norm_type = norm_type
        self.leak_rate = 0.2
        self.activation = nn.LeakyReLU(self.leak_rate, inplace=True)
        self.branch_num = branch_num

        #
        self.input_conv = LayerConv(
            in_channels=in_channels,
            out_channels=self.encoder_list[0],
            layers=self.layers,
            activation=self.activation,
            norm_type=self.norm_type)

        self.downsample1 = DownsampleConv(
            in_channels=self.encoder_list[0],
            out_channels=self.encoder_list[1],
            layers=self.layers,
            downsample_type="maxpool",
            init_type=self.init_type,
            activation=self.activation,
            norm_type=self.norm_type
        )
        self.downsample2 = DownsampleConv(
            in_channels=self.encoder_list[1],
            out_channels=self.encoder_list[2],
            layers=self.layers,
            downsample_type="maxpool",
            init_type=self.init_type,
            activation=self.activation,
            norm_type=self.norm_type
        )

        self.downsample3 = DownsampleConv(
            in_channels=self.encoder_list[2],
            out_channels=self.encoder_list[3],
            layers=self.layers,
            downsample_type="maxpool",
            init_type=self.init_type,
            activation=self.activation,
            norm_type=self.norm_type
        )

        self.mid = MidConv(
            in_channels=self.encoder_list[3],
            out_channels=self.decoder_list[0],
            layers=self.layers,
            downsample_type='maxpool',
            upsample_type='bilinear',
            init_type=self.init_type,
            activation=self.activation,
            norm_type=self.norm_type)

        self.bottom1 = M.conv2d_block(
            in_channels=self.decoder_list[0],
            out_channels=self.decoder_list[0],
            kernel_size=3,
            stride=1,
            padding=1,
            pad_type='reflect',
            init_type=self.init_type,
            activation=self.activation,
            norm_type=self.norm_type
        )

        self.upsample_conv1 = UpsampleConv(
            in_channels=self.decoder_list[0]+self.encoder_list[3],
            out_channels=self.decoder_list[0],
            layers=2,
            upsample_type='bilinear',
            init_type=self.init_type,
            activation=self.activation,
            norm_type=self.norm_type
        )
        self.bottom2 = M.conv2d_block(
            in_channels=self.decoder_list[0],
            out_channels=self.decoder_list[1],
            kernel_size=3,
            stride=1,
            padding=1,
            pad_type='reflect',
            init_type=self.init_type,
            activation=self.activation,
            norm_type=self.norm_type
        )


        self.upsample_conv2 = UpsampleConv(
            in_channels=self.decoder_list[1] + self.encoder_list[2],
            out_channels=self.decoder_list[1],
            layers=self.layers,
            upsample_type='bilinear',
            init_type=self.init_type,
            activation=self.activation,
            norm_type=self.norm_type

        )
        self.bottom3 = M.conv2d_block(
            in_channels=self.decoder_list[1],
            out_channels=self.decoder_list[2],
            kernel_size=3,
            stride=1,
            padding=1,
            pad_type='reflect',
            init_type=self.init_type,
            activation=self.activation,
            norm_type=self.norm_type
        )

        self.upsample_conv3 = UpsampleConv(
            in_channels=self.decoder_list[2] + self.encoder_list[1],
            out_channels=self.decoder_list[2],
            layers=self.layers,
            upsample_type='bilinear',
            init_type=self.init_type,
            activation=self.activation,
            norm_type=self.norm_type
        )
        self.bottom4 = M.conv2d_block(
            in_channels=self.decoder_list[2],
            out_channels=self.decoder_list[3],
            kernel_size=3,
            stride=1,
            padding=1,
            pad_type='reflect',
            init_type=self.init_type,
            activation=self.activation,
            norm_type=self.norm_type
        )

        self.output_conv = LayerConv(
            in_channels=self.decoder_list[3] + self.encoder_list[0],
            out_channels=self.decoder_list[3],
            layers=self.layers,
            init_type='xavier',
            activation=self.activation,
            norm_type=self.norm_type
        )
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.prob_layer = nn.Linear(32, len(branch_num))

        self.output_layer = [MotionBlurArgumentLayer(self.decoder_list[-1], weight_num=x).to(self.device) for x in branch_num]


    @time_log
    def forward(self, inp1, inp2, epsilon):
        '''
        input: NCHW, C:6, H:256, W:256
        output: synthesize image
        '''

        B, C, H, W = inp1.shape

        x = torch.cat([inp1, inp2], dim=1)

        x_conv = self.input_conv(x)
        x1 = self.downsample1(x_conv)
        x2 = self.downsample2(x1)
        x3 = self.downsample3(x2)

        x4 = self.mid(x3)
        x4_b = self.bottom1(x4)
        x4_c = torch.cat([x4_b, x3], dim=1)

        x5 = self.upsample_conv1(x4_c)
        x5_b = self.bottom2(x5)
        x5_c = torch.cat([x5_b, x2], dim=1)

        x6 = self.upsample_conv2(x5_c)
        x6_b = self.bottom3(x6)
        x6_c = torch.cat([x6_b, x1], dim=1)

        x7 = self.upsample_conv3(x6_c)
        x7_b = self.bottom4(x7)
        x7_c = torch.cat([x7_b, x_conv], dim=1)

        x8 = self.output_conv(x7_c)

        prob = self.avg_pool(x8)
        prob = prob.view(B, -1)
        prob = self.prob_layer(prob)
        prob = F.softmax(prob, dim=1)

        random_num = torch.rand(1)
        if random_num.item() < epsilon:
            select_num = int(torch.randint(low=0, high=len(self.branch_num), size=(1,)).item())
            offset1, offset2, weight1, weight2 = self.output_layer[select_num](x8)
            sample = self.line_prediction(
                offset1, offset2, weight1, weight2, inp1, inp2, self.branch_num[select_num])
            return sample
        else:
            select_num = prob.max(dim=1)[1]
            sample_list = []
            for b in range(select_num.shape[0]):
                offset1, offset2, weight1, weight2 = self.output_layer[select_num[b]](x8[b:b+1])
                sample = self.line_prediction(offset1, offset2, weight1, weight2, inp1[b:b+1], inp2[b:b+1], self.branch_num[select_num[b]])
                sample_list.append(sample)
            return torch.cat(sample_list, dim=0)

    @time_log
    def line_prediction(self, offset1, offset2, weight1, weight2, img1, img2, N):
        B, C, H, W = img1.shape
        norm_factor = torch.FloatTensor([H/2, W/2]).to(self.device)
        theta = torch.FloatTensor([1, 0, 0, 0, 1, 0]).view(
            1, 2, 3).repeat(B, 1, 1).to(self.device)
        grid = F.affine_grid(theta, img1.shape)
        grid.mul_(norm_factor).unsqueeze_(0)

        offset1_sample = offset1.permute(
            0, 2, 3, 1).contiguous().unsqueeze(0).repeat(N, 1, 1, 1, 1)
        offset2_sample = offset2.permute(
            0, 2, 3, 1).contiguous().unsqueeze(0).repeat(N, 1, 1, 1, 1)
        sample_factor = torch.FloatTensor(
            [x for x in range(N)]).view(N, 1, 1, 1, 1).to(self.device)

        offset1_sample.mul_(sample_factor).add_(grid).div_(norm_factor)
        offset2_sample.mul_(sample_factor).add_(grid).div_(norm_factor)

        sample1 = []
        sample2 = []
        for n in range(N):
            sample1.append(F.grid_sample(
                img1, offset1_sample[n], padding_mode='border'))
            sample2.append(F.grid_sample(
                img2, offset2_sample[n], padding_mode='border'))

        sample1 = torch.stack(sample1, dim=1)
        sample2 = torch.stack(sample2, dim=1)

        weight1 = weight1.view(B, N, 1, H, W)
        weight2 = weight2.view(B, N, 1, H, W)

        sample = (weight1*sample1 + weight2*sample2).sum(dim=1)

        return sample


def test_synthesize_blur():
    net = SynthesizeBlur()
    print(net)
    inputs = torch.randn(4, 3, 256, 256)
    if torch.cuda.is_available():
        net = net.cuda()
        inputs = inputs.cuda()
    out = net(inputs, inputs)


def test_synthesize_blur_branch():
    net = SynthesizeBlurBranch()
    print(net)
    inputs = torch.randn(4, 3, 128, 128)
    if torch.cuda.is_available():
        net = net.cuda()
        inputs = inputs.cuda()
    epsilon_list = [x*1.0/100 for x in range(100)]
    for item in epsilon_list:
        out = net(inputs, inputs, item)
        print(out.shape)


if __name__ == '__main__':
    test_synthesize_blur_branch()
