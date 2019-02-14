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

from network import ImageInferenceNetwork
from network import GradientInferenceNetwork
from dataset import CrrnDatasetRgb
from loss import CRRN_loss


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


def train(train_dataloader, dev_dataloader, IiN, GiN, optimizer, lr_scheduler, exp_dir, args):
    IiN.train()
    GiN.train()
    if args.loss_function == 'CRRN_loss':
        criterion = CRRN_loss
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
            img, input_GiN, background, reflection, background_gradient =
                img.cuda(), input_GiN.cuda(), background.cuda(), reflection.cuda(), background_gradient.cuda()
            estimate_gradient_B, gradient_guide = GiN(input_GiN)
            estimate_B, estimate_R = IiN(img, gradient_guide)

            loss = criterion(estimate_B, background, estimate_R, reflection, estimate_gradient_B, background_gradient)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print('[epoch%d: batch%d], train loss: %f' % (epoch, idx, loss.item()))

            if (idx+1) % 200 == 0:
                total_count = 0.
                total_loss = 0.
                for index, data in enumerate(dev_dataloader):
                    img, input_GiN, background, reflection, background_gradient = data
                    img, input_GiN, background, reflection, background_gradient =
                        img.cuda(), input_GiN.cuda(), background.cuda(), reflection.cuda(), background_gradient.cuda()
                    estimate_gradient_B, gradient_guide = GiN(input_GiN)
                    estimate_B, estimate_R = IiN(img, gradient_guide)

                    loss = criterion(estimate_B, background, estimate_R, reflection, estimate_gradient_B, background_gradient)
                    total_count += 1
                    total_loss += loss.item()
                print('[epoch{}: batch{}], test avg loss: {}'.format(epoch, idx, total_loss / total_count))
                print('[epoch{}: batch{}], test avg loss: {}'.format(epoch, idx, total_loss / total_count), file=log_f)
        if (epoch+1) % 3 == 0:
            torch.save(model.state_dict(), "%s/epoch_%d.pth" % (exp_dir, epoch))
    log_f.close()


def validate(test_dataloader, IiN, GiN, exp_dir):
    count = 0
    validate_dir = os.path.join(exp_dir, 'validate')
    if not os.exists(validate_dir):
        os.mkdir(validate_dir)
    for index, data in enumerate(test_dataloader):
        img, input_GiN, background, reflection, background_gradient = data
        img, input_GiN, background, reflection, background_gradient =
            img.cuda(), input_GiN.cuda(), background.cuda(), reflection.cuda(), background_gradient.cuda()
        estimate_gradient_B, gradient_guide = GiN(input_GiN)
        estimate_B, estimate_R = IiN(img, gradient_guide)
        B = torch.cat([estimate_B, background], dim=2)
        R = torch.cat([estimate_R, reflection], dim=2)
        I = torch.cat([estimate_B, img], dim=2)
        for t in range(I.shape[0]):
            cv2.imwrite('%s/I%d.jpg' % (validate_dir, count), I.data.cpu().numpy())
            cv2.imwrite('%s/B%d.jpg' % (validate_dir, count), B.data.cpu().numpy())
            cv2.imwrite('%s/R%d.jpg' % (validate_dir, count), R.data.cpu().numpy())
            count += 1


def main(args):
    if args.GiN == 'GradientInferenceNetwork':
        GiN = GradientInferenceNetwork()
    else:
        raise ValueError("input GiN type: {}".format(args.GiN))
    if args.IiN == 'ImageInferenceNetwork':
        IiN = ImageInferenceNetwork(backbone_type='vgg16')
    else:
        raise ValueError("input IiN type: {}".format(args.IiN))
    GiN = GiN.cuda()
    IiN = IiN.cuda()

    parameters = [item for item in IiN.parameters()]
    for item in GiN.parameters():
        parameters.append(item)
    optimizer = torch.optim.Adam([parameters, args.lr, weight_decay=args.weight_decay)
    lr_scheduler = MultiStepLR(optimizer, milestones=[30], gamma=0.1)

    load_model(GiN, args.load_path_GiN, strict=True)
    print('load GiN state dict in {}'.format(args.load_path_GiN))
    if args.load_path_IiN:
        if args.recover:
            load_model(IiN, args.load_path_IiN, strict=True)
            print('load IiN state dict in {}'.format(args.load_path_IiN))

    exp_dir = "%s/%s_%s_lr%f_w%f_b%d" % (args.output_dir, args.model,
              args.loss_function, args.lr, args.weight_decay, args.batch_size)
    if not os.path.exists(exp_dir):
        os.mkdir(exp_dir)

    root = args.root
    transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Resize(args.resize_scale[0])
                    transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))])
    train_set = CrrnDatasetRgb(root=root, train=True, transform=transform)
    test_set = CrrnDatasetRgb(root=root, train=False, transform=transform)
    train_dataloader = DataLoader(train_set, batch_size=args.batch_size,
                                  shuffle=True, num_workers=args.num_workers,
                                  pin_memory=True)
    test_dataloader = DataLoader(test_set, batch_size=args.batch_size,
                                 shuffle=False, num_workers=args.num_workers,
                                 pin_memory=True)

    if args.evaluate:
        validate(test_dataloader, model, exp_dir)
        return

    train(train_dataloader, test_dataloader, IiN, GiN,
          optimizer, lr_scheduler, exp_dir, args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='CRRN joint GiN and IiN')
    parser.add_argument(
        '--load_path_IiN', default='./experiment/', type=str)
    parser.add_argument(
        '--load_path_GiN', default='./experiment/', type=str)
    parser.add_argument('--root', default='./data/')
    parser.add_argument('--recover', default=False, type=bool)
    parser.add_argument('--epoch', default=50, type=int)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--IiN', default='ImageInferenceNetwork', type=str)
    parser.add_argument('--GiN', default='GradientInferenceNetwork', type=str)
    parser.add_argument('--multi_scale', default=False)
    parser.add_argument('--resize_scale',[(224, 288), (96, 160)])
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--evaluate', default=False, type=bool)
    parser.add_argument('--loss_function', default='CRRN_loss', type=str)
    parser.add_argument('--output_dir', default='experimen/RGB_COCO/', type=str)

    args = parser.parse_args()
    print(args)
    main(args)
