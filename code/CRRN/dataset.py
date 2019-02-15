import cv2
import os.path
import torch
from torch.utils.data import Dataset
import numpy as np
from PIL import Image


class CrrnDatasetRgb(Dataset):

    def __init__(self, root, train=True, transform=None):
        self.root = root
        self.train_path = os.path.join(root, 'train.txt')
        self.test_path = os.path.join(root, 'test.txt')
        self.resize_scale = (224, 288)
        self.triplets = []

        # B,R
        if train:
            with open(self.train_path, 'r') as f:
                items = f.readlines()
            for item in items:
                split_items = item[:-1].split(',')
                self.triplets.append(split_items)
        else:
            with open(self.test_path, 'r') as f:
                items = f.readlines()
            for item in items:
                split_items = item[:-1].split(',')
                self.triplets.append(split_items)

    def rgb2gray(self, image):
        return np.dot(image[..., :3], [0.299, 0.587, 0.144])

    def extract_gradient(self, origin, DEBUG=False):
        assert(isinstance(origin, np.ndarray))
        image = np.copy(origin)
        image = cv2.GaussianBlur(image, (3, 3), 0)
        canny = cv2.Canny(image, 50, 150)
        if DEBUG:
            cv2.imshow("canny", canny)
            cv2.waitKey(0)
        return canny

    def open_image(self, index):
        # return: (np.ndarray: B, R)
        background_path = os.path.join(self.root, self.triplets[index][0])
        reflection_path = os.path.join(self.root, self.triplets[index][1])
        background = cv2.imread(background_path)
        reflection = cv2.imread(reflection_path)
        return background, reflection

    def __getitem__(self, index):
        MIN_ALPHA = 0.8
        MAX_ALPHA = 1.0
        MIN_BETA = 0.1
        MAX_BETA = 0.5

        background, reflection = self.open_image(index)
        background = cv2.resize(background, self.resize_scale)
        reflection = cv2.resize(reflection, self.resize_scale)

        alpha = (np.random.randn(1) - MIN_ALPHA) / (MAX_ALPHA - MIN_ALPHA)
        beta = (np.random.randn(1) - MIN_BETA) / (MAX_BETA - MIN_BETA)
        background = (alpha*background).astype(np.uint8)
        reflection = (beta*reflection).astype(np.uint8)
        img = background + reflection
        img_gradient = self.extract_gradient(img)
        H, W, C = background.shape
        background_downsample = cv2.resize(background, (W//4, H//4))
        background_gradient = self.extract_gradient(background_downsample)

        background = torch.from_numpy(background).float().permute(2, 0, 1)
        reflection = torch.from_numpy(reflection).float().permute(2, 0, 1)
        img = torch.from_numpy(img).float().permute(2, 0, 1)
        background_gradient = torch.from_numpy(background_gradient).float().unsqueeze(0)
        img_gradient = torch.from_numpy(img_gradient).float().unsqueeze(0)
        input_GiN = torch.cat([img, img_gradient], dim=0)

        return img, input_GiN, background, reflection, background_gradient

    def __len__(self):
        return len(self.triplets)


if __name__ == "__main__":
    root = './'
    W = 224
    H = 288
    train_set = CrrnDatasetRgb(root=root, train=True)
    print(len(train_set))
    img, input_GiN, background, reflection, background_gradient = train_set[0]
    print('img', img.shape)
    print('input_GiN', img.shape)
    print('background_gradient', background_gradient.shape)
