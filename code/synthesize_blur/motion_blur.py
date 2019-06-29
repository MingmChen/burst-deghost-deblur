import cv2
import time
import random
import glob
from functools import partial
import os
import shutil
from multiprocessing import Pool

input_path = "data/result_frames.txt"
input_test_path = "data/result_frames_test.txt"
output_path = "data/result_frames_output.txt"

img_cache = {}


def get_random_kernel_template(folder):
    filename = random.choice(glob.glob("{}/*.png".format(folder)))
    if filename not in img_cache:
        img = cv2.imread(filename, 0)
        img_cache[filename] = img
    img = img_cache[filename]
    img[img < 20] = 0
    # random resize
    img = cv2.resize(img, (50, 50))
    if random.random() < 0.5:
        img = cv2.flip(img, 1)
    img = cv2.resize(img, None, fx=random.uniform(
        0.8, 1.2), fy=random.uniform(0.8, 1.2))
    # random rotate
    rows, cols = img.shape
    M = cv2.getRotationMatrix2D(
        (cols / 2, rows / 2), random.choice([0, 90, 180, 270]), 1)
    img = cv2.warpAffine(img, M, (cols, rows))
    return img


def _motion_blur(kernel_folder, imgs):
    kernel_motion_blur = get_random_kernel_template(kernel_folder)
    kernel_motion_blur = kernel_motion_blur / kernel_motion_blur.sum()
    # applying the kernel to the input image
    output = []
    for img in imgs:
        kernel = cv2.resize(kernel_motion_blur, None, fx=random.uniform(
            0.95, 1.05), fy=random.uniform(0.95, 1.05))
        rows, cols = kernel.shape
        M = cv2.getRotationMatrix2D(
            (cols / 2, rows / 2), random.choice([0, 5, 10, 15]), 1)
        kernel = cv2.warpAffine(kernel, M, (cols, rows))
        o = cv2.filter2D(img, -1, kernel)

        output.append(o)
    return output


motion_blur = partial(_motion_blur, "kernels")


def _motion_blur_single(kernel_folder, img):
    kernel_motion_blur = get_random_kernel_template(kernel_folder)
    kernel_motion_blur = kernel_motion_blur / kernel_motion_blur.sum()
    kernel = cv2.resize(kernel_motion_blur, None, fx=random.uniform(
        0.95, 1.05), fy=random.uniform(0.95, 1.05))
    rows, cols = kernel.shape
    M = cv2.getRotationMatrix2D(
        (cols / 2, rows / 2), random.choice([0, 5, 10, 15]), 1)
    kernel = cv2.warpAffine(kernel, M, (cols, rows))
    o = cv2.filter2D(img, -1, kernel)
    return img


motion_blur_single = partial(_motion_blur_single, "kernels")


def generate_blur(output_dir, imglist):

    imgs = []
    save_path = ""
    for i, txt in enumerate(imglist):
        if i == 0:
            # use the first iamge name as path_prefix
            save_path = os.path.join(output_dir, txt) + '.output'
            if os.path.exists(save_path) and os.path.isdir(save_path):
                shutil.rmtree(save_path)
                # os.remove(save_path) # del -recurion
            try:
                os.mkdir(save_path)
            except Exception:
                print("[wrong path]:"+save_path)
        img = cv2.imread(txt.strip())
        # print(img.shape)
        imgs.append(img)
    #outputs = pool.map(motion_blur_single, imgs)
    outputs = motion_blur(imgs)
    # print(len(outputs))
    outputs_str = ""
    if os.path.exists(save_path):
        for idx, img in enumerate(outputs):
            path = os.path.join(save_path, "%d.png" % idx)
            cv2.imwrite(path, img)
            outputs_str += path+"\t"
    else:
        raise ValueError

    outputs_str += imglist[3]+'\n'
    return outputs_str


def burst_func(sublist, id=0):
    sublist = [x.strip() for x in sublist]
    # print(sublist)
    root = os.path.dirname(sublist[0])
    output_dir = os.path.join(root, 'motion_blur')
    if not os.path.exists(output_dir):
        try:
            os.mkdir(output_dir)
        except Exception:
            print("[wrong dir]:" + output_dir)
            return "\n"

    output_str = generate_blur(output_dir, sublist)
    return output_str


def main():

    pool = Pool(24)
    input_txt = open(input_path, encoding='utf-8')
    lines = input_txt.readlines()
    N = len(lines)
    lines.sort()
    print(lines[:10])
    str_to_save = []
    task_list = []
    def parent_func(x): return x.split('/')[-2]
    func = partial(burst_func, id=1)
    for idx, each in enumerate(lines):
        if idx > len(lines)-8:
            break
        sublist = lines[idx:idx+8]
        p1, p9 = parent_func(sublist[0]), parent_func(sublist[-1])
        if p1 != p9:
            continue
        task_list.append(sublist)
    for i in range(0, len(task_list), 96):
        strs = pool.map(func, task_list[i:i+96])
        str_to_save += strs
        print("done:{} / {}".format(i, N))
        with open(output_path, "w", encoding='utf-8') as f:
            f.writelines(str_to_save)
    with open(output_path, "w", encoding='utf-8') as f:
        f.writelines(str_to_save)
    pool.close()
    pool.join()


if __name__ == "__main__":
    main()
