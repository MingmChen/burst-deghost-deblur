import os
from PIL import Image
from torch.utils.data import Dataset


class BurstBlurDataset(Dataset):
    def __init__(self, root=None, train=True, transform=None, burst=8):
        assert(root is not None)
        assert(transform is not None)
        self.transform = transform
        if train:
            self.data_path = os.path.join(root, 'train.txt')
        else:
            self.data_path = os.path.join(root, 'test.txt')
        with open(self.data_path, 'r') as f:
            self.data_list = f.readlines()
        self.burst = burst

    def __len__(self):
        return len(self.data_list)

    def open_image(self, path):
        return Image.Open(path).convert('RGB')

    def __getitem__(self, idx):
        item = self.data_list[idx]
        split_item = item[:-1].split('\t')
        data = []
        for i in range(0, self.burst):
            img = self.open_image(split_item[i])
            img = self.transform(img)
            data.append(img)
        burst_img = torch.stack(data, dim=0)

        gt = self.open_image(split_item[-1])
        gt = self.transform(gt)

        return burst_img, gt
