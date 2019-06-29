import numpy as np
import cv2
from functools import partial
from multiprocessing import Pool
import multiprocessing as mp
from run import generate_burst, get_video_interp_model


def generate_train_func(data, model=None):
    MIN_HEIGHT = 270
    MIN_WIDTH = 270
    def preprocess(path):
        img = cv2.imread(path)
        img = cv2.resize(img, (0, 0), fx=0.25, fy=0.25, interpolation=cv2.INTER_CUBIC)
        h = (img.shape[0] - MIN_HEIGHT) // 2
        w = (img.shape[1] - MIN_WIDTH) // 2
        img = img[h:h+MIN_HEIGHT, w:w+MIN_WIDTH]
        return img
    imgs = [preprocess(x) for x in data]
    burst = generate_burst(imgs[:9], model)

    base_path = 'data/video_interp_train_data/' + data[0].split('/')[-2] + data[0].split('/')[-1].split('.')[0]
    output_paths = []
    gt_path = base_path + '_gt.jpg'
    cv2.imwrite(gt_path, imgs[-1])
    for i in range(8):
        output_path = base_path + '_{}.jpg'.format(i)
        output_paths.append(output_path)
        cv2.imwrite(output_path, burst[i])
    output_paths += [gt_path]
    return '\t'.join(output_paths)+'\n'


def main():
    mp.set_start_method('spawn')
    frames_path = 'data/result_frames.txt'
    output_frames_path = 'data/video_interp_train_data.txt'
    with open(frames_path, 'r', encoding='utf-8') as f:
        paths = f.readlines()
    total_len = len(paths)
    paths.sort()
    print(paths[:15])

    parent_func = lambda x : x.split('/')[-2]
    count = 0
    data_list = []
    for i in range(total_len-8):
        gt_path = paths[i+4][:-1]
        parent_dir1, parent_dir9 = parent_func(paths[i][:-1]), parent_func(paths[i+8][:-1])
        if parent_dir1 != parent_dir9:
            continue
        else:
            data_list.append([paths[x][:-1] for x in range(i, i+9)] + [gt_path])

    pool = Pool(8)
    model = get_video_interp_model()
    func = partial(generate_train_func, model=model)
    #'''
    data_train_list = []
    for i in range(0, len(data_list), 100):
        data_train = pool.map(func, data_list[i:i+100])
        data_train_list += data_train
        count += 100
        print(count)
        with open(output_frames_path, 'w', encoding='utf-8') as f:
            f.writelines(data_train_list)
    '''
    data_train_list = []
    for i in range(len(data_list)):
        data_train_list.append(func(data_list[i]))
        if i%50 == 0:
            print(i)
        with open(output_frames_path, 'w', encoding='utf-8') as f:
            f.writelines(data_train_list)
    '''
    print('total count', len(data_train_list))
    with open(output_frames_path, 'w', encoding='utf-8') as f:
        f.writelines(data_train_list)
    print('end')


if __name__ == "__main__":
    main()
