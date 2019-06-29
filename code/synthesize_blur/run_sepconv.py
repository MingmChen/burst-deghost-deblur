#!/usr/bin/env python

import torch

import getopt
import math
import numpy as np
import os
import PIL
import PIL.Image
import sys
import cv2
try:
    from sepconv import sepconv  # the custom separable convolution layer
except:
    sys.path.insert(0, './sepconv')
    import sepconv  # you should consider upgrading python
# end

##########################################################


##########################################################

#arguments_strModel = 'l1'
#arguments_strFirst = './images/first.png'
#arguments_strSecond = './images/second.png'
#arguments_strOut = './out.png'

# for strOption, strArgument in getopt.getopt(sys.argv[1:], '', [ strParameter[2:] + '=' for strParameter in sys.argv[1::2] ])[0]:
#	if strOption == '--model' and strArgument != '': arguments_strModel = strArgument # which model to use, l1 or lf, please see our paper for more details
#	if strOption == '--first' and strArgument != '': arguments_strFirst = strArgument # path to the first frame
#	if strOption == '--second' and strArgument != '': arguments_strSecond = strArgument # path to the second frame
#	if strOption == '--out' and strArgument != '': arguments_strOut = strArgument # path to where the output should be stored
# end

##########################################################

class Network(torch.nn.Module):
    def __init__(self):
        super(Network, self).__init__()

        def Basic(intInput, intOutput):
            return torch.nn.Sequential(
                torch.nn.Conv2d(
                    in_channels=intInput, out_channels=intOutput, kernel_size=3, stride=1, padding=1),
                torch.nn.ReLU(inplace=False),
                torch.nn.Conv2d(
                    in_channels=intOutput, out_channels=intOutput, kernel_size=3, stride=1, padding=1),
                torch.nn.ReLU(inplace=False),
                torch.nn.Conv2d(
                    in_channels=intOutput, out_channels=intOutput, kernel_size=3, stride=1, padding=1),
                torch.nn.ReLU(inplace=False)
            )
        # end

        def Upsample(intInput, intOutput):
            return torch.nn.Sequential(
                torch.nn.Upsample(
                    scale_factor=2, mode='bilinear', align_corners=True),
                torch.nn.Conv2d(
                    in_channels=intOutput, out_channels=intOutput, kernel_size=3, stride=1, padding=1),
                torch.nn.ReLU(inplace=False)
            )
        # end

        def Subnet():
            return torch.nn.Sequential(
                torch.nn.Conv2d(in_channels=64, out_channels=64,
                                kernel_size=3, stride=1, padding=1),
                torch.nn.ReLU(inplace=False),
                torch.nn.Conv2d(in_channels=64, out_channels=64,
                                kernel_size=3, stride=1, padding=1),
                torch.nn.ReLU(inplace=False),
                torch.nn.Conv2d(in_channels=64, out_channels=51,
                                kernel_size=3, stride=1, padding=1),
                torch.nn.ReLU(inplace=False),
                torch.nn.Upsample(
                    scale_factor=2, mode='bilinear', align_corners=True),
                torch.nn.Conv2d(in_channels=51, out_channels=51,
                                kernel_size=3, stride=1, padding=1)
            )
        # end

        self.moduleConv1 = Basic(6, 32)
        self.moduleConv2 = Basic(32, 64)
        self.moduleConv3 = Basic(64, 128)
        self.moduleConv4 = Basic(128, 256)
        self.moduleConv5 = Basic(256, 512)

        self.moduleDeconv5 = Basic(512, 512)
        self.moduleDeconv4 = Basic(512, 256)
        self.moduleDeconv3 = Basic(256, 128)
        self.moduleDeconv2 = Basic(128, 64)

        self.moduleUpsample5 = Upsample(512, 512)
        self.moduleUpsample4 = Upsample(256, 256)
        self.moduleUpsample3 = Upsample(128, 128)
        self.moduleUpsample2 = Upsample(64, 64)

        self.moduleVertical1 = Subnet()
        self.moduleVertical2 = Subnet()
        self.moduleHorizontal1 = Subnet()
        self.moduleHorizontal2 = Subnet()

        #self.load_state_dict(torch.load('./network-' + arguments_strModel + '.pytorch'))
        self.load_state_dict(torch.load(
            './network-' + 'l1' + '.pytorch', map_location='cpu'))
    # end

    def forward(self, tensorFirst, tensorSecond):
        tensorConv1 = self.moduleConv1(
            torch.cat([tensorFirst, tensorSecond], 1))
        tensorConv2 = self.moduleConv2(torch.nn.functional.avg_pool2d(
            input=tensorConv1, kernel_size=2, stride=2))
        tensorConv3 = self.moduleConv3(torch.nn.functional.avg_pool2d(
            input=tensorConv2, kernel_size=2, stride=2))
        tensorConv4 = self.moduleConv4(torch.nn.functional.avg_pool2d(
            input=tensorConv3, kernel_size=2, stride=2))
        tensorConv5 = self.moduleConv5(torch.nn.functional.avg_pool2d(
            input=tensorConv4, kernel_size=2, stride=2))

        tensorDeconv5 = self.moduleUpsample5(self.moduleDeconv5(
            torch.nn.functional.avg_pool2d(input=tensorConv5, kernel_size=2, stride=2)))
        tensorDeconv4 = self.moduleUpsample4(
            self.moduleDeconv4(tensorDeconv5 + tensorConv5))
        tensorDeconv3 = self.moduleUpsample3(
            self.moduleDeconv3(tensorDeconv4 + tensorConv4))
        tensorDeconv2 = self.moduleUpsample2(
            self.moduleDeconv2(tensorDeconv3 + tensorConv3))

        tensorCombine = tensorDeconv2 + tensorConv2

        tensorFirst = torch.nn.functional.pad(input=tensorFirst, pad=[int(math.floor(51 / 2.0)), int(
            math.floor(51 / 2.0)), int(math.floor(51 / 2.0)), int(math.floor(51 / 2.0))], mode='replicate')
        tensorSecond = torch.nn.functional.pad(input=tensorSecond, pad=[int(math.floor(51 / 2.0)), int(
            math.floor(51 / 2.0)), int(math.floor(51 / 2.0)), int(math.floor(51 / 2.0))], mode='replicate')

        tensorDot1 = sepconv.FunctionSepconv(tensorInput=tensorFirst, tensorVertical=self.moduleVertical1(
            tensorCombine), tensorHorizontal=self.moduleHorizontal1(tensorCombine))
        tensorDot2 = sepconv.FunctionSepconv(tensorInput=tensorSecond, tensorVertical=self.moduleVertical2(
            tensorCombine), tensorHorizontal=self.moduleHorizontal2(tensorCombine))

        return tensorDot1 + tensorDot2
    # end
# end


##########################################################

def estimate(tensorFirst, tensorSecond, moduleNetwork):
    assert(tensorFirst.size(1) == tensorSecond.size(1))
    assert(tensorFirst.size(2) == tensorSecond.size(2))

    intWidth = tensorFirst.size(2)
    intHeight = tensorFirst.size(1)

    # while our approach works with larger images, we do not recommend it unless you are aware of the implications
    assert(intWidth <= 1280)
    # while our approach works with larger images, we do not recommend it unless you are aware of the implications
    assert(intHeight <= 720)

    tensorPreprocessedFirst = tensorFirst.cuda().view(1, 3, intHeight, intWidth)
    tensorPreprocessedSecond = tensorSecond.cuda().view(1, 3, intHeight, intWidth)

    intPreprocessedWidth = int(math.floor(51 / 2.0)) + \
        intWidth + int(math.floor(51 / 2.0))
    intPreprocessedHeight = int(math.floor(51 / 2.0)) + \
        intHeight + int(math.floor(51 / 2.0))

    if intPreprocessedWidth != ((intPreprocessedWidth >> 7) << 7):
        intPreprocessedWidth = (((intPreprocessedWidth >> 7) + 1)
                                << 7) - intPreprocessedWidth  # more than necessary
    # end

    if intPreprocessedHeight != ((intPreprocessedHeight >> 7) << 7):
        intPreprocessedHeight = (((intPreprocessedHeight >> 7) + 1)
                                 << 7) - intPreprocessedHeight  # more than necessary
    # end

    tensorPreprocessedFirst = torch.nn.functional.pad(input=tensorPreprocessedFirst, pad=[int(math.floor(51 / 2.0)), int(math.floor(
        51 / 2.0)) + intPreprocessedWidth, int(math.floor(51 / 2.0)), int(math.floor(51 / 2.0)) + intPreprocessedHeight], mode='replicate')
    tensorPreprocessedSecond = torch.nn.functional.pad(input=tensorPreprocessedSecond, pad=[int(math.floor(51 / 2.0)), int(math.floor(
        51 / 2.0)) + intPreprocessedWidth, int(math.floor(51 / 2.0)), int(math.floor(51 / 2.0)) + intPreprocessedHeight], mode='replicate')

    return torch.nn.functional.pad(input=moduleNetwork(tensorPreprocessedFirst, tensorPreprocessedSecond), pad=[0 - int(math.floor(51 / 2.0)), 0 - int(math.floor(51 / 2.0)) - intPreprocessedWidth, 0 - int(math.floor(51 / 2.0)), 0 - int(math.floor(51 / 2.0)) - intPreprocessedHeight], mode='replicate')[0, :, :, :].cpu()
# end


def get_video_interp_model():
    network = Network().eval()
    return network


def generate_burst(burst, network):
    def preprocess(img):
        img = torch.from_numpy(img).float().permute(2, 0, 1)/255
        return img

    network.cuda()
    outputs = []
    burst = [preprocess(x) for x in burst]
    for i in range(len(burst)-1):
        with torch.no_grad():
            output = estimate(burst[i], burst[i+1], network)
        outputs.append(output)

    def postprocess(img):
        return (img.permute(1, 2, 0).numpy()*255).astype(np.uint8)

    outputs = [postprocess(x) for x in outputs]
    return outputs


def generate33(inp1, inp2, inp3, network):
    inp1 = torch.from_numpy(inp1).float().permute(2, 0, 1)/255
    inp2 = torch.from_numpy(inp2).float().permute(2, 0, 1)/255
    inp3 = torch.from_numpy(inp3).float().permute(2, 0, 1)/255

    network.cuda()
    images = []

    def recursion(img1, img3, num):
        if num < 0:
            return
        with torch.no_grad():
            img2 = estimate(img1, img3, network)
        recursion(img1, img2, num-1)
        images.append(img2)
        recursion(img2, img3, num-1)
    # convert to np array, the value of each pixel is between 0.0 and 1.0
    images.append(inp1)
    images.append(inp2)
    images.append(inp3)

    # default input is torch tensor
    recursion(inp1, inp2, 3)
    recursion(inp2, inp3, 3)

    avg = torch.stack(images, dim=0)
    avg = avg.mean(dim=0)

    # print(len(images)) # 33

    # save image , Delete it!
    # for idx, img in enumerate(images):
    #    PIL.Image.fromarray((torch.tensor(img).clamp(0.0, 1.0).np().transpose(1, 2, 0)[:, :, ::-1] *
    #        255.0).astype(np.uint8)).save("%d.png"%idx)
    #PIL.Image.fromarray((torch.tensor(avg).clamp(0.0, 1.0).np().transpose(1, 2, 0)[:, :, ::-1] * 255.0).astype(np.uint8)).save(arguments_strOut)

    return (avg.permute(1, 2, 0).numpy()*255).astype(np.uint8)


##########################################################
if __name__ == '__main__':
    tensorFirst = torch.FloatTensor(np.array(PIL.Image.open(arguments_strFirst))[
                                    :, :, ::-1].transpose(2, 0, 1).astype(np.float32) * (1.0 / 255.0))
    tensorSecond = torch.FloatTensor(np.array(PIL.Image.open(arguments_strSecond))[
                                     :, :, ::-1].transpose(2, 0, 1).astype(np.float32) * (1.0 / 255.0))
    tensorMid = torch.FloatTensor(np.array(PIL.Image.open(arguments_strOut))[
                                  :, :, ::-1].transpose(2, 0, 1).astype(np.float32) * (1.0 / 255.0))

    #tensorOutput = estimate(tensorFirst, tensorSecond)

    generate33(tensorFirst, tensorMid, tensorSecond)

    #PIL.Image.fromarray((tensorOutput.clamp(0.0, 1.0).np().transpose(1, 2, 0)[:, :, ::-1] * 255.0).astype(np.uint8)).save(arguments_strOut)
# end
