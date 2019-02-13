import cv2
import os.path
import torch.utils.data
import numpy as np
import random
import math

dir_data = '../../../CRRN/data/'


class data(torch.utils.data.Dataset):
    def rgb2gray(self, image):
        return np.dot(image[..., :3], [0.299, 0.587, 0.144])

    def gradient(self, image):
        image = cv2.GaussianBlur(image, (3, 3), 0)
        canny = cv2.Canny(image, 50, 150)
        # cv2.imshow("canny", canny)
        # cv2.waitKey(0)
        return canny

    def mix(self, x):
        return cv2.imread(dir_data + 'input{}.jpg'.format(x))

    def tru(self, x):
        return cv2.imread(dir_data + 'truth{}.jpg'.format(x))

    def GT(self, x):
        return gradient(cv2.imread(dir_data + 'truth{}.jpg'.format(x)))

    def __init__(self, root, catfile='cat.txt', npoints=100, train=True, classifiction=False):
        self.npoints = npoints
        self.root = root
        self.category = {}
        self.classification = classifiction

        train_files = os.path.join(root, '../../../CRRN/data/train')
        dev_files = os.path.join(root, '../../../CRRN/data/dev')

        if train:
            # 数据集为train所用的sample
            self.datapath = sorted(os.listdir(train_files))
            self.datapath = [os.path.join(train_files, i) for i in self.datapath]
        else:
            # 数据集为dev所用的sample
            self.datapath = sorted(os.listdir(dev_files))
            self.datapath = [os.path.join(dev_files, i) for i in self.datapath]

    def __getitem__(self, x, batch_size, data_h=224, data_w=288):
        ret = np.empty([batch_size, 4, data_h, data_w])
        for i in range(0, batch_size):
            image_mix = self.mix(i + x)
            image_mix = cv2.resize(image_mix, dsize=(data_w, data_h), fx=1, fy=1)
            image_mix_gra = self.gradient(image_mix)
            for ii in range(0, data_h):
                for j in range(0, data_w):
                    for k in range(0, 3):
                        ret[i][k][ii][j] = image_mix[ii][j][k] / 255
                    ret[i][3][ii][j] = image_mix_gra[ii][j] / 255
        return ret

    def __len__(self):
        return len(self.datapath)


data_ = data('E:/CCTV/burst-deghost-deblur/code/CRRN')
if __name__ == "__main__":
    data_.__getitem__(0, 4)
