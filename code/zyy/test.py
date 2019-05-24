import torch
from torch import nn


def myconv(mat, weight, kernel=3, stride=1, padding=1, paddingmode="reflect"):
    if paddingmode == "reflect":
        temp = nn.ReflectionPad2d(padding)
        mat = temp(mat)

    groups, inchannel, H, W = mat.shape
    outchannel, inchannel, kernel, kernel = weight.shape
    newH, newW = (H - kernel) // stride + 1, (W - kernel) // stride + 1

    weight = weight.reshape(1, outchannel, inchannel, 1, kernel * kernel)
    weight = weight.expand(groups, -1, -1, newH * newW, -1)

    change = nn.Unfold((3, 3), padding=0, stride=stride)
    mat = change(mat)
    mat = mat.reshape(groups, inchannel, -1, mat.shape[2])
    mat = torch.transpose(mat, 2, 3)
    mat = mat.reshape(groups, 1, inchannel, -1, kernel * kernel)
    mat = mat.expand(-1, outchannel, -1, -1, -1)

    ret = weight * mat
    ret = torch.sum(ret, 4)
    ret = torch.sum(ret, 2)
    ret = ret.reshape(groups, -1, newH, newW)
    return ret


if __name__ == '__main__':
    x = torch.randn(2, 3, 3, 5)
    temp = nn.ReflectionPad2d(1)
    y = temp(x)

    kernel = torch.randn(5, 3, 3, 3)
    ret = nn.functional.conv2d(y, kernel, padding=0, stride=2)

    # print(ret.shape)
    print(ret)

    ret2 = myconv(x, kernel, 3, 2, 1, "reflect")

    # print(ret.shape)
    print(ret2)

    print(torch.sum(ret - ret2))
