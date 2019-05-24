import torch
from torch import nn


def myconv(mat, inchannel, outchannel, weight, kernel=3, stride=1, padding=1, paddingmode="reflect"):
	if paddingmode == "reflect":
		temp = nn.ReflectionPad2d(padding)
		mat = temp(mat)

	groups, channel, H, W = mat.shape
	newH, newW = (H + 1 - kernel) // stride, (W + 1 - kernel) // stride
	ret = torch.randn(groups, outchannel, newH, newW)
	for group in range(0, groups):
		for k in range(0, outchannel):
			for i in range(0, newH):
				for j in range(0, newW):
					temp = mat[group:group + 1, :, i:i + kernel, j:j + kernel]
					# print(temp.shape)
					kernelmat = weight[k:k + 1, :, :]
					ret[group][k][i][j] = torch.sum(temp * kernelmat)
	return ret.float()


if __name__ == '__main__':
	x = torch.randn(1, 3, 3, 4)
	temp = nn.ReflectionPad2d(1)
	y = temp(x)

	kernel = torch.randn(4, 3, 3, 3)
	ret = nn.functional.conv2d(y, kernel, padding=0)

	# print(ret.shape)
	print(ret)

	ret2 = myconv(x, 3, 4, kernel, 3, 1, 1, "reflect")

	# print(ret.shape)
	print(ret2)

	print(torch.sum(ret-ret2))