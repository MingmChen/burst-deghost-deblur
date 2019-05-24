#import os
import torch
from torch.utils.data import Dataset
#from torchvision import transforms, utils


class BlurDataset(Dataset):
    def __init__(self, transform = None):
        self.transform = transform


    def __len__(self):
        return  1024


    def __getitem__(self, item):
        sample = {'input1': torch.rand(3,256,256),
                  'input2': torch.rand(3,256,256),
                  'output': torch.rand(3,256,256)}
        return sample





