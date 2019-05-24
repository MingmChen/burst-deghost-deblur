from typing import Any, Union

import torch
from torch import nn


def myconv(mat, weight, kernel=3, stride=1, padding=1, paddingmode="reflect"):
    if paddingmode == "reflect":
        temp = nn.ReflectionPad2d(padding)
        mat = temp(mat)

    groups, inchannel, H, W = mat.shape
    outchannel, inchannel, kernel, kernel = weight.shape
    newH, newW = (H + 1 - kernel) // stride, (W + 1 - kernel) // stride

    weight = weight.reshape(outchannel, inchannel, 1, kernel * kernel)
    weight = weight.expand(-1, -1, newH * newW, -1)

    change = nn.Unfold((newH, newW), padding=0)
    mat = change(mat)
    mat = mat.reshape(groups, inchannel, -1, kernel * kernel)
    mat = mat.expand(outchannel, -1, -1, -1)

    ret = weight * mat
    ret = torch.sum(ret, 3)
    ret = torch.sum(ret, 1)
    ret = ret.reshape(-1, newH, newW)
    return ret


if __name__ == '__main__':
    x = torch.randn(1, 3, 3, 4)
    temp = nn.ReflectionPad2d(1)
    y = temp(x)

    kernel = torch.randn(4, 3, 3, 3)
    ret = nn.functional.conv2d(y, kernel, padding=0)

    # print(ret.shape)
    print(ret)

    ret2 = myconv(x, kernel, 3, 1, 1, "reflect")

    # print(ret.shape)
    print(ret2)

    print(torch.sum(ret - ret2))
