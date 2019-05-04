# -*- coding:utf-8 -*-
'''
version 1: without any encapsulation
@butub
'''
import torch
import torch.nn as nn
import torchvision
import os


class SythesizeBlur(nn.Module):
    def __init__(self):
        super(SythesizeBlur, self).__init__()

        self.LeakRate = 0.2
        self.N = 17 # N evnely-spaced discrete samples

        '''encoder 1'''
        self.conv11 = nn.Conv2d(6, 32, kernel_size=3, stride=1, padding=1)
        self.relu11 = nn.LeakyReLU(self.LeakRate)
        self.conv12 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.relu12 = nn.LeakyReLU(self.LeakRate)
        self.conv13 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.relu13 = nn.LeakyReLU(self.LeakRate)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2,stride=2)

        '''encoder 2'''
        self.conv21 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.relu21 = nn.LeakyReLU(self.LeakRate)
        self.conv22 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.relu22 = nn.LeakyReLU(self.LeakRate)
        self.Conv23 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
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
        self.conv41 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
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
        self.up6 = nn.Upsample(scale_factor=2, mode='bilinear') # align_corners?
        self.conv61 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.relu61 = nn.LeakyReLU(self.LeakRate)
        # skip connect with relu43: Nx256xHxW
        self.conv62 = nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1)
        self.relu62 = nn.LeakyReLU(self.LeakRate)
        self.conv63 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.relu63 = nn.LeakyReLU(self.LeakRate)

        '''decoder 2'''
        self.up7 = nn.Upsample(scale_factor=2, mode='bilinear')
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
        self.offset1 = nn.Conv2d(32, 2, kernel_size=1, stride=1, padding=0) # what about padding?
        self.offset2 = nn.Conv2d(32, 2, kernel_size=1, stride=1, padding=0) # what about padding?
        self.weight1 = nn.Conv2d(32, self.N, kernel_size=1, stride=1, padding=0) # padding?
        self.weight2 = nn.Conv2d(32, self.N, kernel_size=1, stride=1, padding=0) # padding?

        # todo line prediction layer


    def forward(self, x):
        '''
        x: NCHW, C:6, H:256, W:256
        output: synthesize image
        '''
        x = self.relu11( self.conv11(x))
        x = self.relu12( self.conv12(x))
        relu13 = self.relu13( self.conv13(x))
        x = self.maxpool1(relu13)

        x = self.relu21( self.conv21(x))
        x = self.relu22( self.conv22(x))
        relu23 = self.relu23( self.conv23(x))
        x = self.maxpool2(relu23)

        x = self.relu31( self.conv31(x))
        x = self.relu32( self.conv32(x))
        relu33 = self.relu33( self.conv33(x))
        x = self.maxpool3(relu33)

        x = self.relu41( self.conv41(x))
        x = self.relu42( self.conv42(x))
        relu43 = self.relu43( self.conv43(x))
        x = self.maxpool4(relu43)

        x = self.relu51( self.conv51)
        x = self.relu52( self.conv52)
        x = self.relu53( self.conv53)

        x = self.up6(x)
        x = self.relu61(self.conv61(x))
        x = torch.cat([relu43, x], dim=1) # NCHW, 1--> Channel, for keras(NHWC), dim=3
        x = self.relu62( self.conv62(x))
        x = self.relu63( self.conv63(x))

        x = self.up7(x)
        x = self.relu71(self.conv71(x))
        x = torch.cat([relu33, x], dim=1) # skip connect
        x = self.relu72( self.conv72(x))
        x = self.relu73( self.conv73(x))

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

        offset1 = self.offset1(x)
        offset2 = self.offset2(x)
        weight1 = self.weight1(x)
        weight2 = self.weight2(x)

        # todo feed line prediction layer
        pass

if __name__ == '__main__':
    net = SythesizeBlur()
    print(net)