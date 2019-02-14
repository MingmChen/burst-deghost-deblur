import cv2
import os.path
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np
from PIL import Image


class CrrnDatasetRgb(Dataset):

    def __init__(self, root, train=True, transform=None):
        self.root = root
        self.train_path = os.path.join(root, 'train.txt')
        self.test_path = os.path.join(root, 'test.txt')
        if transform is None:
            raise ValueError('please specify image transform')
        else:
            self.transform = transform

        self.triplets = []

        # B,R
        if train:
            with open(self.train_path, 'r') as f:
                items = f.readlines()
            split_items = items[:-1].split(',')
            self.triplets.append(split_items)
        else:
            with open(self.test_path, 'r') as f:
                items = f.readlines()
            split_items = items[:-1].split(',')
            self.triplets.append(split_items)

    def rgb2gray(self, image):
        return np.dot(image[..., :3], [0.299, 0.587, 0.144])

    def extract_gradient(self, origin, DEBUG=False):
        assert(isinstance(origin, torch.Tensor))
        image = origin.numpy()
        image = cv2.GaussianBlur(image, (3, 3), 0)
        canny = cv2.Canny(image, 50, 150)
        if DEBUG:
            cv2.imshow("canny", canny)
            cv2.waitKey(0)
        return torch.from_numpy(canny)

    def open_image(self, index):
        # return: (PIL.Image: B, R)
        background_path = os.path.join(self.root, self.triplets[index][0])
        reflection_path = os.path.join(self.root, self.triplets[index][1])
        background = Image.open(background_path).convert('RGB')
        reflection = Image.open(reflection_path).convert('RGB')
        return background, reflection

    def __getitem__(self, index):
        MIN_ALPHA = 0.8
        MAX_ALPHA = 1.0
        MIN_BETA = 0.1
        MAX_BETA = 0.5

        background, reflection = self.open_image(index)
        background, reflection = self.transform(background), self.transform(reflection)

        alpha = (torch.randn(1) - MIN_ALPHA) - (MAX_ALPHA - MIN_ALPHA)
        beta = (torch.randn(1) - MIN_BETA) - (MAX_BETA - MIN_BETA)
        background *= alpha
        reflection *= beta
        img = background + reflection
        img_gradient = self.extract_gradient(img)
        background_gradient = self.extract_gradient(background)
        input_GiN = torch.cat([img, img_gradient], dim=0)

        return img, input_GiN, background, reflection, background_gradient

    def __len__(self):
        return len(self.triplets)


if __name__ == "__main__":
    root = './'
    W = 224
    H = 288
    transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(W, H),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        ])
    train_set = CrrnDatasetRgb(root=root, train=True, transform=transform)
