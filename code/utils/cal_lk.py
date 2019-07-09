# coding:utf-8
import sys
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
sys.path.append('../synthesize_blur')
import viz_flow as viz

data_dir = "./"

old_path = data_dir + "1.jpeg"
new_path = data_dir + "2.jpeg"

gap = [0, 100]

feature_params = dict(maxCorners=100000,
                      qualityLevel=0.001,
                      minDistance=10,
                      blockSize=7)

lk_params = dict(winSize=(15, 15),
                 maxLevel=3,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))


def find_last(s, t):
        start = 0
        ret = -1
        while True:
            result = s.find(t, start)
            if result != -1:
                ret = result
            else:
                break
            start = result + 1
        return ret


def viz_lk(flow, path):
    H, W, _ = flow.shape
    f = np.abs(flow[:, :, 0]) + np.abs(flow[:, :, 1])
    f = (f - f.min()) / (f.max() - f.min()+1e-8)
    f *= 255
    f = cv2.GaussianBlur(f, (11, 11), 0)
    f = torch.FloatTensor(f).view(1, 1, H, W)
    f = F.max_pool2d(f, kernel_size=7, stride=7)
    f = f.squeeze().numpy()
    f = cv2.GaussianBlur(f, (5,5), 0)
    plt.imshow(f)
    plt.savefig(path)
    #plt.show()
    plt.clf()

    return f


def cal_lk(old_path, new_path):
    old_img = cv2.imread(old_path)
    H, W, _ = old_img.shape
    old_gray = cv2.cvtColor(old_img, cv2.COLOR_BGR2GRAY)
    new_img = cv2.imread(new_path)
    new_gray = cv2.cvtColor(new_img, cv2.COLOR_BGR2GRAY)
    print(old_img.shape)

    color = np.random.randint(0, 255, (100000, 3))
    mask = np.zeros_like(old_img)

    p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)
    p1, st, err = cv2.calcOpticalFlowPyrLK(
        old_gray, new_gray, p0, None, **lk_params)

    # select good points
    good_new = p1[st == 1]
    good_old = p0[st == 1]
    good = good_new - good_old

    flow = np.zeros((H, W, 2))
    for i, (new, old) in enumerate(zip(good_new, good_old)):
        a, b = new.ravel()
        c, d = old.ravel()
        gapx = abs(a-c)
        gapy = abs(b-d)
        if gapx >= gap[0] and gapy >= gap[0] and gapx <= gap[1] and gapy <= gap[1]:
            mask = cv2.line(mask, (a, b), (c, d), color[i].tolist(), 2)
            frame = cv2.circle(new_img, (a, b), 5, color[i].tolist(), -1)
            flow[int(d)-1, int(c)-1] = np.array((d-b, c-a))
    img = cv2.add(frame, mask)

    flag = find_last(new_path, '.')
    path = new_path[:flag] + '_flow.png'
    viz_lk(flow, path)

    # plot
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    #'''
    plt.subplot(221)
    plt.imshow(img)
    plt.subplot(222)
    plt.imshow(mask,  cmap=plt.cm.gray)
    plt.subplot(223)
    #'''

    good_x = good[:, 0]
    good_y = good[:, 1]
    good_x = good_x.clip(good_x.mean()-3*good_x.std(), good_x.mean()+3*good_x.std())
    good_y = good_y.clip(good_y.mean()-3*good_y.std(), good_y.mean()+3*good_y.std())
    n1, bins1, patches1 = plt.hist(
        good_x, bins=100, range=(good_x.min(), good_x.max()), density=False, facecolor='green', alpha=0.75)
    plt.subplot(224)
    n2, bins2, patches2 = plt.hist(
        good_y, bins=100, range=(good_y.min(), good_y.max()), density=False, facecolor='blue', alpha=0.75)
    plt.savefig(new_path[:flag] + '_total.png')
    #'''
    width = 0.02
    fig, ax = plt.subplots()
    n1 /= n1.sum()
    n2 /= n2.sum()

    rects1 = ax.bar(bins1[:-1], n1, width,
                    label="x displacement", lw=1, alpha=0.4, facecolor='orange')
    rects2 = ax.bar(bins2[:-1], n2, width,
                    label="y displacement", lw=1, alpha=0.4, facecolor='green')
    bin_min = min(bins1.min(), bins2.min())
    rects3 = ax.bar(bin_min, 0.3, width, lw=1, facecolor='blue', alpha=0.4, label='reference')
    plt.legend(loc="upper left")
    #'''
    #plt.show()
    plt.savefig(new_path[:flag] + '_hist.png')
    plt.clf()


def main(data_list):
    with open(data_list, 'r') as f:
        paths = f.readlines()
    count = 0
    for pair in paths:
        old_path, new_path = pair[:-1].split(' ')
        cal_lk(old_path, new_path)
        print(old_path)
        count += 1


if __name__ == '__main__':
    #data_list = './motion_trans.txt'
    data_list = './motion.txt'
    main(data_list)
