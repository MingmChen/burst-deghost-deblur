import cv2
import os
import numpy as np


def generate(img_path, output_path, generate_per_img=5):
    img_list = [item for item in os.listdir(img_path)]
    length = len(img_list)
    items = []
    for i in range(length):
        print(i)
        tot = 0
        if len(items) > 5000:
            break
        for j in range(generate_per_img):
            content = img_path + img_list[i] + ','
            random_val = np.random.randint(0, length - 1)
            img1 = cv2.imread(img_path + img_list[i])
            img2 = cv2.imread(img_path + img_list[random_val])
            a = np.sum(img1)
            b = np.sum(img2)
            # 两张图片亮度差距不要太大。（0.6,0.8）的话在本地跑100张要半分钟
            if a / b < 0.8 and a / b > 0.6:
                content += img_path + img_list[random_val] + '\n'
                items.append(content)
            else:
                generate_per_img += 1
                tot += 1
            # 防止死循环
            if tot > 5: break
    print(len(items))
    with open(output_path, 'w') as f:
        f.writelines(items)


def main():
    # coco_val_path = '/mnt/lustre/share/DSK/datasets/mscoco2017/val2017/'
    # coco_test_path = '/mnt/lustre/share/DSK/datasets/mscoco2017/test2017/'
    coco_test_path = '../../../CRRN/coco/images/test2017/'
    coco_val_path = '../../../CRRN/coco/images/val2017/'
    output_train = './train.txt'
    output_test = './test.txt'
    generate(coco_test_path, output_train)
    generate(coco_val_path, output_test)


if __name__ == "__main__":
    main()
