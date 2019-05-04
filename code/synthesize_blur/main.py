#-*- coding:utf-8 -*-

import torch
from torch.utils.data import DataLoader
from network import SythesizeBlur
from dataset import BlurDataset
import torch.optim as optim

def main():
    print("in")
    net = SythesizeBlur()
    blur_dataset = BlurDataset()
    dataloader = DataLoader(blur_dataset, batch_size= 16)

    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    print("len data:", dataloader.__len__())
    for idx, dict in enumerate(dataloader):
        print("inin")
        input1 , input2, output = dict['input1'], dict['input2'], dict['output']
        sample = net(input1,input2)
        loss = torch.sum(torch.abs(sample - output))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if (idx + 1) % 100 == 0:
            print("idx-{},loss:{}".format(idx+1, loss))




if __name__ == '__main__':
    main()