# -*- coding:utf-8 -*-
'''
version 1: without any encapsulation
@butub
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
sys.path.append('../')
from utils.log import time_log
dtype = torch.cuda.FloatTensor
dtype_long = torch.cuda.LongTensor


@time_log
def bilinear_interpolate_torch(im, x, y):
    x0 = torch.floor(x).type(dtype_long)
    x1 = x0 + 1

    y0 = torch.floor(y).type(dtype_long)
    y1 = y0 + 1

    x0 = torch.clamp(x0, 0, im.shape[1] - 1)
    x1 = torch.clamp(x1, 0, im.shape[1] - 1)
    y0 = torch.clamp(y0, 0, im.shape[0] - 1)
    y1 = torch.clamp(y1, 0, im.shape[0] - 1)

    Ia = im[y0, x0][0]
    Ib = im[y1, x0][0]
    Ic = im[y0, x1][0]
    Id = im[y1, x1][0]

    wa = (x1.type(dtype) - x) * (y1.type(dtype) - y)
    wb = (x1.type(dtype) - x) * (y - y0.type(dtype))
    wc = (x - x0.type(dtype)) * (y1.type(dtype) - y)
    wd = (x - x0.type(dtype)) * (y - y0.type(dtype))

    return torch.t((torch.t(Ia) * wa)) + torch.t(torch.t(Ib) * wb) + torch.t(torch.t(Ic) * wc) + torch.t(torch.t(Id) * wd)


class SythesizeBlur(nn.Module):
    def __init__(self):
        super(SythesizeBlur, self).__init__()
        if torch.cuda.is_available():
            self.device = 'cuda'
        else:
            self.device = 'cpu'

        self.LeakRate = 0.2
        self.N = 17  # N evnely-spaced discrete samples

        '''encoder 1'''
        self.conv11 = nn.Conv2d(6, 32, kernel_size=3, stride=1, padding=1)
        self.relu11 = nn.LeakyReLU(self.LeakRate)
        self.conv12 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.relu12 = nn.LeakyReLU(self.LeakRate)
        self.conv13 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.relu13 = nn.LeakyReLU(self.LeakRate)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        '''encoder 2'''
        self.conv21 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.relu21 = nn.LeakyReLU(self.LeakRate)
        self.conv22 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.relu22 = nn.LeakyReLU(self.LeakRate)
        self.conv23 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.relu23 = nn.LeakyReLU(self.LeakRate)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        '''encoder 3'''
        self.conv31 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.relu31 = nn.LeakyReLU(self.LeakRate)
        self.conv32 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.relu32 = nn.LeakyReLU(self.LeakRate)
        self.conv33 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.relu33 = nn.LeakyReLU(self.LeakRate)
        self.maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        '''encoder 4'''
        self.conv41 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.relu41 = nn.LeakyReLU(self.LeakRate)
        self.conv42 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.relu42 = nn.LeakyReLU(self.LeakRate)
        self.conv43 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.relu43 = nn.LeakyReLU(self.LeakRate)
        self.maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        '''encoder 5'''
        self.conv51 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.relu51 = nn.LeakyReLU(self.LeakRate)
        self.conv52 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.relu52 = nn.LeakyReLU(self.LeakRate)
        self.conv53 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.relu53 = nn.LeakyReLU(self.LeakRate)

        '''decoder 1'''
        self.up6 = nn.Upsample(
            scale_factor=2, mode='bilinear', align_corners=False)
        self.conv61 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.relu61 = nn.LeakyReLU(self.LeakRate)
        # skip connect with relu43: Nx256xHxW
        self.conv62 = nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1)
        self.relu62 = nn.LeakyReLU(self.LeakRate)
        self.conv63 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.relu63 = nn.LeakyReLU(self.LeakRate)

        '''decoder 2'''
        self.up7 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv71 = nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1)
        self.relu71 = nn.LeakyReLU(self.LeakRate)
        # skip connect with relu33:Nx128xHxW
        self.conv72 = nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1)
        self.relu72 = nn.LeakyReLU(self.LeakRate)
        self.conv73 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.relu73 = nn.LeakyReLU(self.LeakRate)

        '''decoder 3'''
        self.up8 = nn.Upsample(scale_factor=2, mode='bilinear')
        # skip connect with relu23
        self.conv81 = nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1)
        self.relu81 = nn.LeakyReLU(self.LeakRate)
        # skip connect with relu23:Nx64xHxW
        self.conv82 = nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1)
        self.relu82 = nn.LeakyReLU(self.LeakRate)
        self.conv83 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.relu83 = nn.LeakyReLU(self.LeakRate)

        '''decoder 4'''
        self.up9 = nn.Upsample(scale_factor=2, mode='bilinear')
        self.conv91 = nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1)
        self.relu91 = nn.LeakyReLU(self.LeakRate)
        # skip connect with relu13:Nx32xHxW
        self.conv92 = nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1)
        self.relu92 = nn.LeakyReLU(self.LeakRate)
        self.conv93 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.relu93 = nn.LeakyReLU(self.LeakRate)

        '''
        outputs:  feeds into line prediction layer
        '''
        self.offset1 = nn.Conv2d(32, 2, kernel_size=1,
                                 stride=1, padding=0)  # what about padding?
        self.offset2 = nn.Conv2d(32, 2, kernel_size=1,
                                 stride=1, padding=0)  # what about padding?
        self.weight1 = nn.Conv2d(
            32, self.N, kernel_size=1, stride=1, padding=0)  # padding?
        self.weight2 = nn.Conv2d(
            32, self.N, kernel_size=1, stride=1, padding=0)  # padding?

        # todo line prediction layer

    @time_log
    def forward(self, inp1, inp2):
        '''
        input: NCHW, C:6, H:256, W:256
        output: synthesize image
        '''

        self.shape = inp1.shape
        self.minibatch = inp1.shape[0]

        x = torch.cat([inp1, inp2], dim=1)
        x = self.relu11(self.conv11(x))
        x = self.relu12(self.conv12(x))
        relu13 = self.relu13(self.conv13(x))
        x = self.maxpool1(relu13)

        x = self.relu21(self.conv21(x))
        x = self.relu22(self.conv22(x))
        relu23 = self.relu23(self.conv23(x))
        x = self.maxpool2(relu23)

        x = self.relu31(self.conv31(x))
        x = self.relu32(self.conv32(x))
        relu33 = self.relu33(self.conv33(x))
        x = self.maxpool3(relu33)

        x = self.relu41(self.conv41(x))
        x = self.relu42(self.conv42(x))
        relu43 = self.relu43(self.conv43(x))
        x = self.maxpool4(relu43)

        x = self.relu51(self.conv51(x))
        x = self.relu52(self.conv52(x))
        x = self.relu53(self.conv53(x))

        x = self.up6(x)
        x = self.relu61(self.conv61(x))
        # NCHW, 1--> Channel, for keras(NHWC), dim=3
        x = torch.cat([relu43, x], dim=1)
        x = self.relu62(self.conv62(x))
        x = self.relu63(self.conv63(x))

        x = self.up7(x)
        x = self.relu71(self.conv71(x))
        x = torch.cat([relu33, x], dim=1)  # skip connect
        x = self.relu72(self.conv72(x))
        x = self.relu73(self.conv73(x))

        x = self.up8(x)
        x = self.relu81(self.conv81(x))
        x = torch.cat([relu23, x], dim=1)  # skip connect
        x = self.relu82(self.conv82(x))
        x = self.relu83(self.conv83(x))

        x = self.up9(x)
        x = self.relu91(self.conv91(x))
        x = torch.cat([relu13, x], dim=1)  # skip connect
        x = self.relu92(self.conv92(x))
        x = self.relu93(self.conv93(x))

        offset1 = self.offset1(x)   # Nx2xHxW
        offset2 = self.offset2(x)   # Nx2xHxW
        weight1 = self.weight1(x)   # Nx17xHxW
        weight2 = self.weight2(x)   # Nx17xHxW

        # todo feed line prediction layer
        sample = self.violent_cycle(
            offset1, offset2, weight1, weight2, inp1, inp2)
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
            sample_n1 = nn.functional.grid_sample(_inp1, grid1, padding_mode='border')
            sample += (_weight1[:, n, :, :]).unsqueeze(1) * sample_n1

        for n in range(self.N):
            grid2 = torch.clamp(grid + (n / self.N) *
                                (_offset2.permute(0, 2, 3, 1)), -1, 1)
            grid2.div_(norm_factor)
            sample_n2 = nn.functional.grid_sample(_inp2, grid2, padding_mode='border')
            sample += (_weight2[:, n, :, :]).unsqueeze(1) * sample_n2

        return sample

    @time_log
    def line_prediction(self, offset1, offset2, weight1, weight2, img1, img2):
        N = self.N
        B, C, H, W = img1.shape
        norm_factor = torch.FloatTensor([H/2, W/2]).to(self.device)
        theta = torch.FloatTensor([1, 0, 0, 0, 1, 0]).view(1, 2, 3).repeat(B, 1, 1).to(self.device)
        grid = F.affine_grid(theta, self.shape)
        grid.mul_(norm_factor).unsqueeze_(0)

        offset1_sample = offset1.permute(0, 2, 3, 1).contiguous().unsqueeze(0).repeat(N, 1, 1, 1, 1)
        offset2_sample = offset2.permute(0, 2, 3, 1).contiguous().unsqueeze(0).repeat(N, 1, 1, 1, 1)
        sample_factor = torch.FloatTensor([x for x in range(N)]).view(N, 1, 1, 1, 1).to(self.device)

        offset1_sample.mul_(sample_factor).add_(grid).div_(norm_factor)
        offset2_sample.mul_(sample_factor).add_(grid).div_(norm_factor)

        sample1 = []
        sample2 = []
        for n in range(N):
            sample1.append(F.grid_sample(img1, offset1_sample[n], padding_mode='border'))
            sample2.append(F.grid_sample(img2, offset2_sample[n], padding_mode='border'))

        sample1 = torch.stack(sample1, dim=1)
        sample2 = torch.stack(sample2, dim=1)

        weight1 = weight1.view(B, N, 1, H, W)
        weight2 = weight2.view(B, N, 1, H, W)

        sample = (weight1*sample1 + weight2*sample2).sum(dim=1)

        return sample


def test_synthesize_blur():
    net = SythesizeBlur()
    inputs = torch.randn(4, 3, 256, 256)
    if torch.cuda.is_available():
        net = net.cuda()
        inputs = inputs.cuda()
    out = net(inputs, inputs)


if __name__ == '__main__':
    test_synthesize_blur()
