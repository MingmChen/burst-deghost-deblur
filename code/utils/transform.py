import torch
import torch.nn as nn
import torch.nn.functional as F

import cv2
import numpy as np


def MaxPool(y):
    x = torch.FloatTensor(y)
    x = x.permute(2, 0, 1).unsqueeze(0)
    x = F.max_pool2d(x, kernel_size=2, stride=2)
    x = x.squeeze(0).permute(1, 2, 0).numpy()
    return x


def AvgPool(y):
    x = torch.FloatTensor(y)
    x = x.permute(2, 0, 1).unsqueeze(0)
    x = F.avg_pool2d(x, kernel_size=2, stride=2)
    x = x.squeeze(0).permute(1, 2, 0).numpy()
    return x


def transform(img_path):
    root = img_path.split('.')[0]
    img = cv2.imread(img_path)
    transform_dict = {
        'maxpool': MaxPool,
        'avgpool': AvgPool,
        'resize_bilinear': lambda x: cv2.resize(x, (0, 0), fx=0.5, fy=0.5, interpolation=cv2.INTER_LINEAR),
        'resize_nearest': lambda x: cv2.resize(x, (0, 0), fx=0.5, fy=0.5, interpolation=cv2.INTER_NEAREST)
    }
    save_path = [root+'_'+x+'.jpg' for x in transform_dict.keys()]
    for idx, data in enumerate(transform_dict.items()):
        _, func = data
        result = func(img)
        cv2.imwrite(save_path[idx], result)


def main(data_list):
    with open(data_list, 'r') as f:
        data = f.readlines()
    for idx, data in enumerate(data):
        transform(data[:-1])
        print(idx)


if __name__ == "__main__":
    data_list = 'trans.txt'
    main(data_list)
