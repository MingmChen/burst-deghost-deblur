import os
import cv2
import torch
import argparse
from network import ImageInferenceNetwork
from network import GradientInferenceNetwork
from dataset import data
import loss


def train():
    for epoch_i in epoch:
        total_loss = 0.0
        for i in range(data_size // bachsize):
            x = i * bachsize
            inputs = dataset.unit_inpus()


def main(args):
    if args.GiN == 'GradientInferenceNetwork':
        GiN = GradientInferenceNetwork()
    else:
        raise ValueError

    if args.IiN == 'ImageInferenceNetwork':
        IiN = ImageInferenceNetwork(backbone_type='vgg16_bn')
    GiN.cuda()
    IiN.cuda()

    if args.load_path:
        if args.recover:
            load_GiN(GiN, args.load_path, strict=True)
            load_IiN(IiN, args.load_path, strict=True)
            print('load model state dict in {}'.format(args.load_path))

    batch_size=args.batch_size
    print('batch_size:{}'.format(batch_size))

    loss_list=[]
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='CRRN')
    parser.add_argument('--load_path', default=None, type=str)
    parser.add_argument('--recover', default=False, type=bool)
    parser.add_argument('--epoch', default=4, type=int)
    parser.add_argument('--lr', default=1e-3, type=float)
    parser.add_argument('--weight_decay', default=1e-5, type=float)
    parser.add_argument('--batch_size', default=4, type=int)
    parser.add_argument('--GiN', default='GradientInferenceNetwork', type=str)
    parser.add_argument('--IiN', default='ImageInferenceNetwork', type=str)
    parser.add_argument('--num_workers', default=2, type=int)
    parser.add_argument('--evaluate', default=False, type=bool)
    parser.add_argument('--loss_function_GiN', default='SILoss', type=str)
    parser.add_argument('--loss_function_IiN', default='SSIMLoss', type=str)
    parser.add_argument('--output_dir', default='./output', type=str)

    args = parser.parse_args()
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)
    print(args)
    main(args)
