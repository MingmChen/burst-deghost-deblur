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

from dataset import BlurDataset # assume dataset name is BlurDataset with data: NxCxHxW , label
#from network import BurstDeblurMP
from network.burst_deblur_network import BurstDeblurMP
from network.loss import GradientLoss

def train(network_model, train_loader, test_loader, optimizer, scheduler, args):
    network_model.train()
    model_dir = os.path.join(args.log_model_dir, '{}'.format(args.exp_name))
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)
    log = open(os.path.join(model_dir, 'log.txt'), 'w')
    print(args, file=log)

    criterion_l1 = nn.L1loss(reduction="mean")
    criterion_gradient = GradientLoss(alpha=1.0)

    global_cnt = 0
    for epoch in range(args.epoch):
        scheduler.step() # update learning rate
        for idx, data in enumerate(train_loader):
            global_cnt += 1
            img, gt = data

            if torch.cuda.is_available():
                img, gt = img.cuda(), gt.cuda()
            output = network_model(img)

            loss = criterion_l1(output, gt)/10 + criterion_gradient(output, gt)
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

    #if args.loss == 'l1_loss': # todo
    #    criterion = nn.L1Loss(reduction='mean') # none | mean | sum
    #else:
    #    raise ValueError

    def lr_func(epoch):
        lr_factor = args.lr_factor_dict
        lr_key = list(lr_factor.keys())
        index = 0
        for i in range(len(lr_key)):
            if epoch < lr_key[i]:
                index = i
                break
        return lr_factor[lr_key[index]]

    optimizer = torch.optim.Adam(
        network_model.parameters(), lr=args.init_lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_func)

    transform = transforms.Compose([ # todo
        transforms.RandomCrop(256),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(90),
        transforms.ToTensor(),

    ])

    train_set = BlurDataset(root=args.root, train=True, transform=transform)
    test_set = BlurDataset(root=args.root, train=False, transform=transform)
    
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
    parser.add_argument('--root', default='/mnt/lustre/niuyazhe/data/syn_data')
    parser.add_argument('--log_model_dir', default='./train_log')
    parser.add_argument('--batch_size', default=32)
    parser.add_argument('--num_workers', default=3)
    #parser.add_argument('--loss', default='l1_loss')
    parser.add_argument('--norm_type', default=None)
    parser.add_argument('--init_lr', default=2e-5)
    #parser.add_argument('--lr_factor_dict', default={15: 1, 40: 0.1, 60: 0.05})
    parser.add_argument('--lr_factor_dict', default={0:1})
    parser.add_argument('--weight_decay', default=1e-5)
    parser.add_argument('--epoch', default=100)
    parser.add_argument('--evaluate', default=False)
    parser.add_argument('--show_interval', default=100)
    parser.add_argument('--test_interval', default=2)
    parser.add_argument('--snapshot_interval', default=1)
    parser.add_argument('--exp_name', 'syn_baseline')
    args = parser.parse_args()
    if not os.path.exists(args.log_model_dir):
        os.mkdir(args.log_model_dir)
    print(args)
    main(args)
