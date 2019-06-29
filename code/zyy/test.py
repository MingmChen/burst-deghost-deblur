import torch
from torch import nn
import cv2
from cv2 import CV_32F
import numpy as np


def myconv(mat, weight, kernel=3, stride=1, padding=1, paddingmode="reflect"):
	mat = nn.functional.pad(mat, (padding, padding, padding, padding), paddingmode, 0)

	groups, inchannel, H, W = mat.shape
	outchannel, inchannel, kernel, kernel = weight.shape
	newH, newW = (H - kernel) // stride + 1, (W - kernel) // stride + 1

	weight = weight.reshape(1, outchannel, inchannel, 1, kernel * kernel)
	weight = weight.expand(groups, -1, -1, newH * newW, -1)

	change = nn.Unfold((kernel, kernel), padding=0, stride=stride)
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


def lp_pooling_unfold(mat, stride, p, padding=0, paddingmode="reflect"):
	mat = nn.functional.pad(mat, (padding, padding, padding, padding), paddingmode, 0)

	groups, inchannel, H, W = mat.shape
	newH, newW = H // stride, W // stride

	change = nn.Unfold((stride, stride), padding=0, stride=stride)
	mat = change(mat)
	mat = mat.reshape(groups, inchannel, -1, mat.shape[2])
	mat = torch.transpose(mat, 2, 3)
	mat = mat ** p

	gaussx = cv2.getGaussianKernel(stride, 1, CV_32F)
	gaussy = cv2.getGaussianKernel(stride, 1, CV_32F)
	gauss_ = gaussx * np.transpose(gaussy)
	gauss = torch.from_numpy(gauss_)
	gauss = gauss.reshape(1, 1, 1, stride * stride)
	gauss = gauss.expand(groups, inchannel, mat.shape[2], -1)
	mat = mat * gauss
	mat = torch.sum(mat, 3)
	mat = mat ** (1 / p)
	mat = mat.reshape(groups, inchannel, newH, newW)
	return mat


def lp_pooling_conv(mat, stride, p, padding=0, paddingmode="reflect"):
	gaussx = cv2.getGaussianKernel(stride, 1, CV_32F)
	gaussy = cv2.getGaussianKernel(stride, 1, CV_32F)
	gauss_ = gaussx * np.transpose(gaussy)
	gauss = torch.from_numpy(gauss_)
	gauss = gauss.reshape(1, 1, stride, stride)

	groups, inchannel, H, W = mat.shape
	mat = mat.reshape(-1, 1, H, W)
	mat = mat ** p
	ret = myconv(mat, gauss, stride, stride, padding, paddingmode)
	ret = ret ** (1/p)
	ret = ret.reshape(groups, inchannel, ret.shape[2], ret.shape[3])
	return ret


if __name__ == '__main__':
	x = torch.randn(2, 3, 3, 5)
	temp = nn.ReflectionPad2d(1)
	y = temp(x)

	kernel = torch.randn(5, 3, 3, 3)
	ret = nn.functional.conv2d(y, kernel, padding=0, stride=2)

	# print(ret.shape)
	#print(ret)

	ret2 = myconv(x, kernel, 3, 2, 1, "reflect")

	# print(ret.shape)
	#print(ret2)

	#print(torch.sum(ret - ret2))

	x = torch.randn(2, 3, 4, 6)
	ret3 = lp_pooling_unfold(x, 2, 2, padding=1)
	print(ret3)

	ret4 = lp_pooling_conv(x, 2, 2, padding=1)
	print(ret4)

	print(torch.sum(ret3-ret4))
