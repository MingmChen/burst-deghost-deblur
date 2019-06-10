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
from dataset import BlurDataset
from network import SynthesizeBlur


class Normalize(object):
    def __init__(self, min_val=0, max_val=255):
        self.min_val = min_val
        self.max_val = max_val

    def __call__(self, x):
        return (x - self.min_val)*1.0 / (self.max_val - self.min_val)


def train(network_model, train_loader, test_loader, optimizer, scheduler, criterion, args):
    network_model.train()
    model_dir = os.path.join(args.log_model_dir, '{}'.format(args.exp_name))
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)
    log = open(os.path.join(model_dir, 'log.txt'), 'w')
    print(args, file=log)

    global_cnt = 0
    for epoch in range(args.epoch):
        scheduler.step()
        for idx, data in enumerate(train_loader):
            global_cnt += 1
            img1, img2, gt = data

            if torch.cuda.is_available():
                img1, img2, gt = img1.cuda(), img2.cuda(), gt.cuda()
            output = network_model(img1, img2)

            loss = criterion(output, gt)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if global_cnt % args.show_interval == 0:
                print(
                    '[epoch:{}, batch:{}]\t'.format(epoch, idx),
                    '[loss: {:.5f}]\t'.format(loss.item()),
                    '[lr: {:.6f}]'.format(scheduler.get_lr()[0])
                )
                print(
                    '[epoch:{}, batch:{}]\t'.format(epoch, idx),
                    '[loss: {:.5f}]\t'.format(loss.item()),
                    '[lr: {:.6f}]'.format(scheduler.get_lr()[0]),
                    file=log
                )
        if epoch % args.test_interval == 0:
            loss_sum = 0.
            test_batch_num = 0
            total_num = 0
            for idx, data in enumerate(test_loader):
                test_batch_num += 1
                img1, img2, gt = data
                total_num += img1.shape[0]

                if torch.cuda.is_available():
                    img1, img2, gt = img1.cuda(), img2.cuda(), gt.cuda()
                output = network_model(img1, img2)

                loss = criterion(output, gt)

                loss_sum += loss.item()
            print('\n***************validation result*******************')
            print(
                'loss_avg: {:.5f}\t'.format(loss_sum / test_batch_num),
            )
            print('****************************************************\n')
            print('\n***************validation result*******************', file=log)
            print(
                'loss_avg: {:.5f}\t'.format(loss_sum / test_batch_num),
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

    network_model = SynthesizeBlur(norm_type=args.norm_type)

    if torch.cuda.is_available():
        network_model = network_model.cuda()
        network_model = torch.nn.DataParallel(network_model)

    if args.loss == 'l1_loss':
        criterion = nn.L1Loss() # none | mean | sum
    else:
        raise ValueError

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

    transform = transforms.Compose([
        transforms.CenterCrop(256),
        #transforms.RandomHorizontalFlip(p=0.5),
        #transforms.RandomVerticalFlip(p=0.5),
        #transforms.RandomRotation(90),
        transforms.ToTensor(),
        Normalize()
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
          test_loader, optimizer, scheduler, criterion, args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', default='/mnt/lustre/niuyazhe/data/syn_data_sort')
    parser.add_argument('--log_model_dir', default='./train_log')
    parser.add_argument('--batch_size', default=16)
    parser.add_argument('--num_workers', default=3)
    parser.add_argument('--loss', default='l1_loss')
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
    parser.add_argument('--exp_name', default='syn_baseline')
    args = parser.parse_args()
    if not os.path.exists(args.log_model_dir):
        os.mkdir(args.log_model_dir)
    print(args)
    main(args)
