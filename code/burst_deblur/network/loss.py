import torch
import torch.nn as nn
import torch.nn.functional as F


class GradientLoss(nn.Module):
    #grad_kernel_type_list = ['sobel']

    def __init__(self, grad_kernel_type='scharr', size_average=True):
        # assert(grad_kernel_type in GradientLoss.grad_kernel_type_list)
        super(GradientLoss, self).__init__()
        if grad_kernel_type == 'sobel':
            self.grad_cal = self._sobel
        elif grad_kernel_type == 'scharr':
            self.grad_cal = self._scharr
        else:
            raise ValueError
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.size_average = size_average

    def _sobel(self, img):
        assert(isinstance(img, torch.Tensor))
        assert(len(img.shape) == 4)
        C = img.shape[1]
        assert(C in [1, 3])
        kernel_x = torch.FloatTensor(
            [-1, 0, 1, -2, 0, 2, -1, 0, 1]).reshape(1, 1, 3, 3).repeat(C, 1, 1, 1).to(self.device)
        kernel_y = torch.FloatTensor(
            [-1, -2, -1, 0, 0, 0, 1, 2, 1]).reshape(1, 1, 3, 3).repeat(C, 1, 1, 1).to(self.device)
        grad_x = F.conv2d(img, kernel_x, stride=1, padding=1, groups=C)
        grad_y = F.conv2d(img, kernel_y, stride=1, padding=1, groups=C)
        return torch.abs(grad_x) + torch.abs(grad_y)

    def _scharr(self, img):
        assert(isinstance(img, torch.Tensor))
        assert(len(img.shape) == 4)
        C = img.shape[1]
        assert(C in [1, 3])
        kernel_x = torch.FloatTensor(
            [-3, 0, 3, -10, 0, 10, -3, 0, 3]).reshape(1, 1, 3, 3).repeat(C, 1, 1, 1).to(self.device)
        kernel_y = torch.FloatTensor(
            [-3, -10, -3, 0, 0, 0, 3, 10, 3]).reshape(1, 1, 3, 3).repeat(C, 1, 1, 1).to(self.device)
        grad_x = F.conv2d(img, kernel_x, stride=1, padding=1, groups=C)
        grad_y = F.conv2d(img, kernel_y, stride=1, padding=1, groups=C)
        return torch.abs(grad_x) + torch.abs(grad_y)

    def forward(self, x, gt):
        x_grad = self.grad_cal(x)
        gt_grad = self.grad_cal(gt)
        loss = torch.abs(x_grad-gt_grad)
        if self.size_average:
            loss = loss.mean()
        else:
            loss = loss.sum()
        return loss


def test_grad_loss():
    inputs = torch.randn(4, 1, 32, 32)
    loss = GradientLoss(alpha=1.0)
    loss_val = loss(inputs, inputs-0.1)
    print(loss_val.item())


if __name__ == "__main__":
    test_grad_loss()
