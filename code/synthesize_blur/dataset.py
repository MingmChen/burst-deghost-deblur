import os
from PIL import Image
from torch.utils.data import Dataset


class BlurDataset(Dataset):
    def __init__(self, root=None, train=True, transform=None):
        assert(root is not None)
        assert(transform is not None)
        self.transform = transform
        if train:
            self.data_path = os.path.join(root, 'train.txt')
        else:
            self.data_path = os.path.join(root, 'test.txt')
        with open(self.data_path, 'r') as f:
            self.data_list = f.readlines()

    def __len__(self):
        return len(self.data_list)

    def open_image(self, path):
        return Image.open(path).convert('RGB')

    def __getitem__(self, idx):
        item = self.data_list[idx]
        split_item = item[:-1].split('\t')
        data = []
        for path in split_item:
            img = self.open_image(path)
            img = self.transform(img)
            data.append(img)

        return data
