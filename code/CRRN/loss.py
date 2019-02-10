import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class GaussKernel(object):
    '''
        only for 1D or 2D output
        return: 2D tensor kernel
    '''
    def __init__(self, kernel_size, sigma, channel):
        self.kernel_size = kernel_size
        self.sigma = sigma
        self.channel = channel

    def get_kernel_1d(self, is_output=False):
        radius = self.kernel_size // 2
        val = [np.exp(- (x-radius)**2 / (2.0*(self.sigma**2))) for x in range(self.kernel_size)]
        kernel = torch.Tensor(val).float()
        if is_output:
            kernel = kernel / kernel.sum()
            kernel = kernel.unsqueeze(0).unsqueeze(0)
            kernel = kernel.expand(self.channel, 1, self.kernel_size)
            return kernel
        else:
            return kernel / kernel.sum()

    def get_kernel_2d(self):
        kernel_1d = self.get_kernel_1d(is_output=False).unsqueeze(1)
        kernel_2d = kernel_1d.mm(kernel_1d.t()).float().unsqueeze(0).unsqueeze(0)
        kernel_2d = kernel_2d.expand(self.channel, 1, self.kernel_size, self.kernel_size).contiguous()
        return kernel_2d


class SILoss(nn.Module):
    def __init__(self,
                 channel,
                 size_average=True,
                 kernel_size=11,
                 sigma=1.5,
                 max_val=255,
                 k3=0.03/2):
        super(SILoss, self).__init__()
        self.channel = channel
        self.size_average = size_average
        self.max_val = max_val
        self.kernel_size = kernel_size
        self.sigma = sigma
        self.k3 = k3

    def _get_kernel(self, img):
        assert(len(img.shape) == 4)
        b, c, h, w = img.shape
        real_size = min(h, w, self.kernel_size)
        real_sigma = real_size * self.sigma / self.kernel_size

        gauss_handle = GaussKernel(real_size, real_sigma, self.channel)
        kernel = gauss_handle.get_kernel_2d()
        return kernel, real_size

    def __call__(self, img1, img2):
        kernel, real_kernel_size = self._get_kernel(img1)
        if torch.cuda.is_available() and isinstance(img1, torch.cudaTensor):
            kernel = kernel.cuda()
        padding = real_kernel_size // 2
        groups = self.channel

        sigma_11 = F.conv2d(img1*img1, kernel, padding=padding, groups=groups)
        sigma_12 = F.conv2d(img1*img2, kernel, padding=padding, groups=groups)
        sigma_22 = F.conv2d(img2*img2, kernel, padding=padding, groups=groups)

        c3 = (self.k3 * self.max_val)**2
        val1 = 2.0 * sigma_12 + c3
        val2 = sigma_11 + sigma_22 + c3
        si_metric = val1 / val2
        if self.size_average:
            return 1 - si_metric.mean()
        else:
            # TODO(test validation)
            return -si_metric.sum()


def ssim_loss(estimate, gt, kernel_size=1):
    ssim_val = 0
    return 1 - ssim_val


def total_loss(estimate_B, gt_B, estimate_R, gt_R, estimate_g_B, gt_g_B):
    B_FACTOR = 0.8
    l1_criterion = nn.L1Loss()
    si_criterion = SILoss(channel=1, max_val=1.0)
    ret = (B_FACTOR * ssim_loss(estimate_B, gt_B) +
           l1_criterion(estimate_B, gt_B) +
           ssim_loss(estimate_R, gt_R) +
           si_criterion(estimate_g_B, gt_g_B))
    return ret


def unit_test_gausskernel():
    kernel_size = 5
    sigma = 1.0
    channel = 3
    handle = GaussKernel(kernel_size, sigma, channel)
    res = handle.get_kernel_2d()
    print(res.shape)
    print(res)


def unit_test_SILoss():
    #label = torch.randn(4, 3, 224, 224)
    label = torch.from_numpy(np.load('loss_test.npy')).float()
    pred = label.clone()
    pred[:, 0, 0, :] = 1.0
    if torch.cuda.is_available():
        label = label.cuda()
        pred = pred.cuda()

    criterion = SILoss(channel=3, max_val=1.0)
    loss = criterion(pred, label)
    print(loss.item())


if __name__ == "__main__":
    # unit_test_gausskernel()
    unit_test_SILoss()
