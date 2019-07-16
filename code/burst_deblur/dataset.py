import os
import os.path as op
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from PIL import Image


class CocoBlurDataset(Dataset):
    def __init__(self,
                 root_dir,
                 transform,
                 burst_num=8):

        self.root_dir = root_dir
        self.meta_file = op.join(root_dir, 'meta.txt')
        self.transform = transform
        self.burst_num = burst_num

        with open(self.meta_file, 'r') as f:
            data = f.readlines()
        self.img_ids = [x[:-1] for x in data]

    def __len__(self):
        return len(self.img_ids)

    def _open_image(self, path):
        return Image.open(path).convert('RGB')

    def __getitem__(self, idx):
        img_id = self.img_ids[idx]
        prefix = op.join(self.root_dir, img_id.split('.')[0])

        gt = prefix + '_gt.png'
        burst = [prefix + '_{}.png'.format(x+1) for x in range(self.burst_num)]
        gt = self.transform(self._open_image(gt))
        burst = [self.transform(self._open_image(x)) for x in burst]

        burst = torch.stack(burst, dim=0)

        return burst, gt
