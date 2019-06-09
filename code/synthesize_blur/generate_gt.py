import numpy as np
import cv2
from multiprocessing import Pool


def generate_func(triplet):
    MIN_HEIGHT = 270
    MIN_WIDTH = 270
    img1 = cv2.imread(triplet[0])
    img2 = cv2.imread(triplet[1])
    img3 = cv2.imread(triplet[2])
    if img1.shape[0] < MIN_HEIGHT or img1.shape[1] < MIN_WIDTH:
        return "ERROR"
    img1 = cv2.resize(img1, 0, fx=0.25, fy=0.25, interpolation=cv2.INTER_CUBIC)
    img2 = cv2.resize(img2, 0, fx=0.25, fy=0.25, interpolation=cv2.INTER_CUBIC)
    img3 = cv2.resize(img3, 0, fx=0.25, fy=0.25, interpolation=cv2.INTER_CUBIC)
    h = (img1.shape[0] - MIN_HEIGHT) // 2
    w = (img1.shape[1] - MIN_WIDTH) // 2
    img1 = img1[h:h+MIN_HEIGHT, w:w+MIN_WIDTH]
    img2 = img2[h:h+MIN_HEIGHT, w:w+MIN_WIDTH]
    img3 = img3[h:h+MIN_HEIGHT, w:w+MIN_WIDTH]
    gt = func(img1, img2, img3)

    base_path = '/mnt/lustre/niuyazhe/data/syn_data/' + triplet[0].split('/')[-2] + triplet[0].split('/')[-1].split('.')[0]
    img1_path = base_path + '1.jpg'
    img3_path = base_path + '2.jpg'
    gt_path = base_path + 'gt.jpg'
    return img1_path+'\t'+img3_path+'\t'+gt_path+'\n'


def main():
    frames_path = '~/data/result_frames_test.txt'
    output_frames_path = '~/data/syn_train_data.txt'
    with open(frames_path, 'r', encoding='utf-8') as f:
        paths = f.readlines()
    total_len = len(paths)

    parent_func = lambda x : x.split('/')[-2]
    count = 0
    triplet_list = []
    for i in range(total_len-2):
        path1 = paths[i]
        path2 = paths[i+1]
        path3 = paths[i+2]
        parent_dir1, parent_dir3 = parent_func(path1), parent_func(path3)
        if parent_dir1 != parent_dir3:
            continue
        else:
            triplet_list.append([path1, path2, path3])

    pool = Pool(24)
    triplet_train_list = pool.map(generate_func, triplet_list)
    with open(output_frames_path, 'w', encoding='utf-8') as f:
        f.writelines(triplet_train_list)
    print('end')


if __name__ == "__main__":
    main()
