import torch
import torch.nn as nn


def ssim_loss(estimate, gt):
    return 1 - ssim_val


def si_loss(estimate, gt):
    return 1 - si_val

def total_loss(estimate_B, gt_B, estimate_R, gt_R, estimate_g_B, gt_g_B):
    B_FACTOR = 0.8
    ret = B_FACTOR * ssim_loss(estimate_B, gt_B) +
          nn.L1Loss(estimate_B, gt_B) +
          ssim_loss(estimate_R, gt_R) +
          si_loss(estimate_g_B, gt_g_B)
    return ret
