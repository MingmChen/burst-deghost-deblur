import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import reduce
import matplotlib.pyplot as plt
import sys
sys.path.append('../base_model')
from resnet import resnet18, resnet50


def extract_feature(model, inputs, layer_list):
    feature_map_list = []
    handle_list = []

    def hook(module, input_feature, output_feature):
        feature_map_list.append(output_feature)

    for name in layer_list:
        layer = getattr(model, name)
        handle = layer.register_forward_hook(hook)
        handle_list.append(handle)
    with torch.no_grad():
        _ = model(inputs)
    for item in handle_list:
        item.remove()
    return feature_map_list


def resnet_feature(net, img1, img2):
    layer_list = ['maxpool', 'layer1', 'layer2', 'layer3', 'layer4', 'avgpool', 'fc']
    img1 = torch.FloatTensor(img1).permute(2, 0, 1).unsqueeze(0)
    img2 = torch.FloatTensor(img2).permute(2, 0, 1).unsqueeze(0)
    img1 = (img1 - img1.min()) / (img1.max() - img1.min())
    img2 = (img2 - img2.min()) / (img2.max() - img2.min())
    img1_feature_list = extract_feature(net, img1, layer_list)
    img2_feature_list = extract_feature(net, img2, layer_list)
    layer_list.insert(0, 'input')
    img1_feature_list.insert(0, img1)
    img2_feature_list.insert(0, img2)
    layer_list.append('softmax')
    img1_feature_list.append(F.softmax(img1_feature_list[-1], dim=1))
    img2_feature_list.append(F.softmax(img2_feature_list[-1], dim=1))

    dist_func = lambda x: torch.norm(torch.abs(x), p=1)

    def relative_error1(data):
        rel_avg_dist = (torch.abs(data[0] - data[1]))/(1e-6+torch.abs(data[0]))
        return rel_avg_dist.clamp(0, 1)

    def relative_error2(data):
        mask = torch.where(torch.abs(data[0]) > 1e-6, torch.ones_like(data[0]), torch.zeros_like(data[1])).byte()
        data0 = torch.masked_select(data[0], mask)
        data1 = torch.masked_select(data[1], mask)
        result = torch.abs(data0 - data1) / torch.abs(data0)
        return result.clamp(0, 1)

    for idx, data in enumerate(zip(img1_feature_list, img2_feature_list)):
        dist = dist_func(data[0] - data[1])
        shape_mul = reduce(lambda x, y: x*y, data[0].shape)
        rel_avg_dist1 = relative_error1(data)
        rel_avg_dist2 = relative_error2(data)
        print('layer:{}\tavg L1 dist:{:.6f}\tavg rel L1 dist:{:.4f}\tavg rel L1 dist(nonzero):{:.4f}\tmax val dist:{:.4f}'.format(
            layer_list[idx], dist/shape_mul,
            rel_avg_dist1.mean(), rel_avg_dist2.mean(),
            torch.abs(data[0] - data[1]).max()
        ))

    return
    fc1 = img1_feature_list[-1].squeeze().numpy()
    fc2 = img2_feature_list[-1].squeeze().numpy()
    fig, ax = plt.subplots()
    bins = np.arange(0, 1000, 1)
    width = 1
    ax.bar(bins, fc1, width, lw=1, facecolor='orange', alpha=1.0)
    ax.bar(bins, fc2, width, lw=1, facecolor='green', alpha=0.2)
    plt.show()


def main(data_list):
    with open(data_list, 'r') as f:
        data = f.readlines()
    net = resnet18(pretrained=True)
    for item in data:
        img1_path, img2_path = item[:-1].split(' ')
        img1 = cv2.imread(img1_path)
        img2 = cv2.imread(img2_path)
        resnet_feature(net, img1, img2)
        print(img1_path)


if __name__ == "__main__":
    path = 'motion_trans_imagenet.txt'
    main(path)
