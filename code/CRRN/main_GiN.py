import os
import argparse
from collections import OrderedDict
import numpy as np
import torch
import torch.nn as nn
import torch.optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import MultiStepLR
from torchvision import transforms

from network import GradientInferenceNetwork
from dataset import CrrnDatasetRgb
from loss import GiN_loss


def load_model(model, path, strict=False):
    state_dict = torch.load(path)
    new_state_dict = OrderedDict()
    for key, val in state_dict.items():
        if key[:6] == 'module':
            name = key[7:]  # remove 'module'
        else:
            name = key
        new_state_dict[name] = val
    model.load_state_dict(new_state_dict, strict=strict)
    all_keys = set(new_state_dict.keys())
    actual_keys = set(model.state_dict().keys())
    missing_keys = actual_keys - all_keys
    for k in missing_keys:
        print(k)


def train(train_dataloader, dev_dataloader, GiN, optimizer, lr_scheduler, exp_dir, args):
    GiN.train()
    if args.loss_function == 'GiN_loss':
        criterion = GiN_loss
    else:
        raise ValueError("invalid lossfunction: {}".format(args.loss_function))
    batch_size = args.batch_size
    log_f = open("%s/log.txt" % (exp_dir), "w")
    print(args, file=log_f)

    for epoch in range(args.epoch):
        lr_scheduler.step()
        current_lr = lr_scheduler.get_lr()[0]
        print('current_lr: {}'.format(current_lr))
        print('current_lr: {}'.format(current_lr), file=log_f)
        for idx, data in enumerate(train_dataloader):
            img, input_GiN, background, reflection, background_gradient = data
            input_GiN, background_gradient = input_GiN.cuda(), background_gradient.cuda()
            estimate_gradient_B, gradient_guide = GiN(input_GiN)

            loss = criterion(estimate_gradient_B, background_gradient)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print('[epoch%d: batch%d], train loss: %f' % (epoch, idx, loss.item()))

            if (idx+1) % 200 == 0:
                total_count = 0.
                total_loss = 0.
                for index, data in enumerate(dev_dataloader):
                    img, input_GiN, background, reflection, background_gradient = data
                    input_GiN, background_gradient = input_GiN.cuda(), background_gradient.cuda()
                    estimate_gradient_B, gradient_guide = GiN(input_GiN)

                    loss = criterion(estimate_gradient_B, background_gradient)
                    total_count += 1
                    total_loss += loss.item()
                print('[epoch{}: batch{}], test avg loss: {}'.format(epoch, idx, total_loss / total_count))
                print('[epoch{}: batch{}], test avg loss: {}'.format(epoch, idx, total_loss / total_count), file=log_f)
        if (epoch+1) % 3 == 0:
            torch.save(GiN.state_dict(), "%s/GiN_epoch_%d.pth" % (exp_dir, epoch))
    log_f.close()


def validate(test_dataloader, GiN, exp_dir):
    GiN.eval()
    count = 0
    validate_dir = os.path.join(exp_dir, 'validate')
    if not os.exists(validate_dir):
        os.mkdir(validate_dir)
    for index, data in enumerate(test_dataloader):
        img, input_GiN, background, reflection, background_gradient = data
        input_GiN, background_gradient = input_GiN.cuda(), background_gradient.cuda()
        with torch.no_grad():
            estimate_gradient_B, gradient_guide = GiN(input_GiN)
        B_g = torch.cat([estimate_gradient_B, background_gradient], dim=2)
        for t in range(I.shape[0]):
            cv2.imwrite('%s/B_gradient%d.jpg' % (validate_dir, count), B_g.data.cpu().numpy())
            count += 1


def main(args):
    if args.GiN == 'GradientInferenceNetwork':
        GiN = GradientInferenceNetwork()
    else:
        raise ValueError("input GiN type: {}".format(args.GiN))
    GiN = GiN.cuda()
    GiN.to_cuda()

    optimizer = torch.optim.Adam(GiN.parameters(), args.lr, weight_decay=args.weight_decay)
    lr_scheduler = MultiStepLR(optimizer, milestones=[40], gamma=0.1)

    if args.load_path_GiN:
        if args.recover:
            load_model(GiN, args.load_path_GiN, strict=True)
            print('load GiN state dict in {}'.format(args.load_path_GiN))

    exp_dir = "%s/%s_%s_lr%f_w%f_b%d" % (args.output_dir, args.GiN,
              args.loss_function, args.lr, args.weight_decay, args.batch_size)
    if not os.path.exists(exp_dir):
        os.mkdir(exp_dir)

    root = args.root
    train_set = CrrnDatasetRgb(root=root, train=True)
    test_set = CrrnDatasetRgb(root=root, train=False)
    train_dataloader = DataLoader(train_set, batch_size=args.batch_size,
                                  shuffle=True, num_workers=args.num_workers,
                                  pin_memory=True)
    test_dataloader = DataLoader(test_set, batch_size=args.batch_size,
                                 shuffle=False, num_workers=args.num_workers,
                                 pin_memory=True)

    if args.evaluate:
        validate(test_dataloader, GiN, exp_dir)
        return

    train(train_dataloader, test_dataloader, GiN,
          optimizer, lr_scheduler, exp_dir, args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='CRRN GiN')
    parser.add_argument(
        '--load_path_GiN', default='./experiment/', type=str)
    parser.add_argument('--root', default='./')
    parser.add_argument('--recover', default=False, type=bool)
    parser.add_argument('--epoch', default=40, type=int)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--GiN', default='GradientInferenceNetwork', type=str)
    parser.add_argument('--multi_scale', default=False)
    parser.add_argument('--resize_scale',default=[(224, 288), (96, 160)])
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--evaluate', default=False, type=bool)
    parser.add_argument('--loss_function', default='GiN_loss', type=str)
    parser.add_argument('--output_dir', default='experiment/RGB_COCO/', type=str)

    args = parser.parse_args()
    print(args)
    main(args)
