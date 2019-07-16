import os
import os.path as op
from functools import partial
from multiprocessing import Pool

import cv2
import numpy as np
cv2.ocl.setUseOpenCL(False)

import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from pycocotools import mask as maskUtils
from pycocotools.coco import COCO
from data_preprocess.motion_blur_kernel import adaptive_blur_kernel, apply_blur


class CocoDataset(Dataset):
    def __init__(self,
                 root_dir,
                 anno_file):

        self.root_dir = root_dir
        self.meta_file = anno_file

        self.coco = COCO(anno_file)
        category_ids = self.coco.cats.keys()

        self.img_ids = list(set([_['image_id'] for _ in self.coco.anns.values()]))

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        img_id = self.img_ids[idx]
        meta_img = self.coco.imgs[img_id]
        filename = os.path.join(self.root_dir, meta_img['file_name'])

        img = cv2.imread(filename, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        return {'img': img, 'filename':meta_img['file_name']}


def single_image_burst(inputs, output_dir, burst_num=8):
    gt = inputs['img']
    filename = inputs['filename']
    cv2.imwrite(os.path.join(output_dir, filename.split('.')[0]+'_gt.png'), gt)

    for i in range(burst_num):
        kernel = adaptive_blur_kernel(gt)
        if kernel is None:
            with open(os.path.join(output_dir, 'error.txt'), 'a') as f:
                f.writelines(filename+'\n')
            print('invalid img name:{}, size:{}'.format(filename, gt.shape))
            return False
        blur_img = apply_blur(gt, kernel)
        cv2.imwrite(os.path.join(output_dir, filename.split('.')[0]+'_{}.png'.format(i+1)), blur_img)
    return True


def generate_burst_blur_data():
    output_dir = 'coco_burst_deblur_train'
    root_dir = 'mscoco2017/train2017'
    anno_file = 'mscoco2017/annotations/instances_train2017.json'
    coco_dataset = CocoDataset(root_dir, anno_file)

    if not op.exists(output_dir):
        os.mkdir(output_dir)

    pool = Pool(24)
    func = partial(single_image_burst, output_dir=output_dir)
    img_list = []
    temp_list = []
    print(len(coco_dataset))
    for idx, data in enumerate(coco_dataset):
        temp_list.append(data)
        if len(temp_list) == 24:
            handle = pool.map(func, temp_list)
            for i, flag in enumerate(handle):
                if flag:
                    img_list.append(temp_list[i]['filename']+'\n')
            temp_list = []
            print(idx)
    pool.close()
    with open(op.join(output_dir, 'meta.txt'), 'w') as f:
        f.writelines(img_list)
    print('finish')


if __name__ == "__main__":
    generate_burst_blur_data()
