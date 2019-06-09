#!/usr/bin/env python
# -*- coding: utf-8 -*-
# =======================================
# File Name : train.py
# Purpose : build a compute graph
# Creation Date : 2019-03-23 23:28
# Last Modified :
# Created By : niuyazhe
# =======================================


import os
import sys
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
sys.path.append('../')

from dataset import BurstBlurDataset # assume dataset name is BlurDataset with data: NxCxHxW , label
#from network import BurstDeblurMP
from network.burst_deblur_network import BurstDeblurMP
from network.loss import GradientLoss


class L1GradLoss(nn.Module):
    def __init__(self, alpha=0.1):
        super(L1GradLoss, self).__init__()
        self.L1 = nn.L1Loss()
        self.grad = GradientLoss(alpha)

    def forward(self, x, gt):
        return self.L1(x, gt) + self.grad(x, gt)


class AddNoise(object):
    def __init__(self, sigma_read=1e-5, sigma_shot=4e-3):
        self.sigma_read = sigma_read
        self.sigma_shot = sigma_shot

    def __call__(self, x):
        assert(isinstance(x, torch.Tensor))
        variance = torch.ones(x.shape) * self.sigma_read**2
        variance += self.sigma_shot*x
        std = torch.sqrt(variance)
        x = torch.normal(x, std)
        return x


def train(network_model, train_loader, test_loader, optimizer, scheduler, criterion, args):
    network_model.train()
    model_dir = os.path.join(args.log_model_dir, '{}'.format(args.exp_name))
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)
    log = open(os.path.join(model_dir, 'log.txt'), 'w')
    print(args, file=log)


    global_cnt = 0
    for epoch in range(args.epoch):
        scheduler.step() # update learning rate
        for idx, data in enumerate(train_loader):
            global_cnt += 1
            img, gt = data

            if torch.cuda.is_available():
                img, gt = img.cuda(), gt.cuda()
            output = network_model(img)

            loss = criterion(output, gt)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if global_cnt % args.show_interval == 0:
                print(
                    '[epoch:{}, batch:{}]\t'.format(epoch, idx),
                    '[loss: {:.3f}]\t'.format(loss.item()),
                    '[lr: {:.6f}]'.format(scheduler.get_lr()[0])
                )
                print(
                    '[epoch:{}, batch:{}]\t'.format(epoch, idx),
                    '[loss: {:.3f}]\t'.format(loss.item()),
                    '[lr: {:.6f}]'.format(scheduler.get_lr()[0]),
                    file=log
                )
        if epoch % args.test_interval == 0:
            loss_sum = 0.
            test_batch_num = 0
            total_num = 0
            for idx, data in enumerate(test_loader):
                test_batch_num += 1
                img, gt = data
                total_num += img.shape[0]

                if torch.cuda.is_available():
                    img, gt = img.cuda(), gt.cuda()
                output = network_model(img)

                loss = criterion(output, gt)

                loss_sum += loss.item()
            print('\n***************validation result*******************')
            print(
                'loss_avg: {:.3f}\t'.format(loss_sum / test_batch_num),
            )
            print('****************************************************\n')
            print('\n***************validation result*******************', file=log)
            print(
                'loss_avg: {:.3f}\t'.format(loss_sum / test_batch_num),
                file=log
            )
            print('****************************************************\n', file=log)

        if epoch % args.snapshot_interval == 0:
            torch.save(network_model.state_dict(), os.path.join(
                model_dir, 'epoch-{}.pth'.format(epoch+1)))
    torch.save(network_model.state_dict(), os.path.join(model_dir, 'epoch-final{}.pth'.format(args.epoch)))
    log.close()


def test(network_model, test_loader):
    raise NotImplementedError


def main(args):

    is_train = (args.evaluate == True)

    network_model = BurstDeblurMP()

    if torch.cuda.is_available():
        network_model = network_model.cuda()

    if args.loss == 'l1_loss': # todo
        criterion = nn.L1Loss(reduction='mean') # none | mean | sum
    elif args.loss == 'l1_grad_loss':
        criterion = L1GradLoss()
    else:
        raise ValueError

    def lr_func(epoch):
        return np.pow(0.999997, epoch)

    optimizer = torch.optim.Adam(
        network_model.parameters(), lr=args.init_lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_func)

    transform = transforms.Compose([ # todo
        transforms.ToTensor(),
        transforms.normalize(mean=(0.5,0.5), std=(0.5,0.5)),
        AddNoise()
    ])

    train_set = BurstBlurDataset(root=args.root, train=True, transform=transform)
    test_set = BurstBlurDataset(root=args.root, train=False, transform=transform)

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers, pin_memory=True)

    if args.evaluate:
        test(network_model, test_loader)
        return

    train(network_model, train_loader,
          test_loader, optimizer, scheduler, args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', default='/mnt/lustre/niuyazhe/data/burst_deblur_data')
    parser.add_argument('--log_model_dir', default='./train_log')
    parser.add_argument('--batch_size', default=32)
    parser.add_argument('--num_workers', default=3)
    parser.add_argument('--loss', default='l1_grad_loss')
    parser.add_argument('--norm_type', default=None)
    parser.add_argument('--init_lr', default=3e-4)
    parser.add_argument('--weight_decay', default=1e-5)
    parser.add_argument('--epoch', default=1000)
    parser.add_argument('--evaluate', default=False)
    parser.add_argument('--show_interval', default=100)
    parser.add_argument('--test_interval', default=2)
    parser.add_argument('--snapshot_interval', default=1)
    parser.add_argument('--exp_name', 'burst_deblur_baseline')
    args = parser.parse_args()
    if not os.path.exists(args.log_model_dir):
        os.mkdir(args.log_model_dir)
    print(args)
    main(args)
