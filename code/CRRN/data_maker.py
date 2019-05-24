from pycocotools.coco import COCO
import cv2
import random
import numpy as np
import sys

sys.path.append(
    'E:\CCTV\CRRN\coco\PythonAPI')

dataDir = 'E:/CCTV/CRRN/coco'
dataType = 'val2017'
annFile = '{}/annotations/instances_{}.json'.format(dataDir, dataType)
coco = COCO(annFile)

#######################################################################合并两个图
'''
0.8660348529
0.9579585346
0.9920034515
'''
div = [0.8660348529, 0.9579585346, 0.9920034515]


def combine(image_a, image_b):
    image_a, image_b = cut(image_a, image_b)
    ###############生成两个图的亮度差距函数ssim_l和亮度大小比较
    mu1 = np.sum(image_a) / len(image_a)
    mu2 = np.sum(image_b) / len(image_b)
    ssim_l = 2 * mu1 * mu2 / (mu1 * mu1 + mu2 * mu2)
    com = 1 if mu1 < mu2 else 0
    ###############
    x = random.uniform(0.8, 1)
    if ssim_l < div[0]:
        if com == 1:
            y = random.uniform(0.1, 0.15)
        else:
            y = random.uniform(0.45, 0.5)
    elif ssim_l < div[1]:
        if com == 1:
            y = random.uniform(0.15, 0.2)
        else:
            y = random.uniform(0.4, 0.45)
    elif ssim_l < div[2]:
        if com == 1:
            y = random.uniform(0.2, 0.25)
        else:
            y = random.uniform(0.35, 0.4)
    else:
        if com == 1:
            y = random.uniform(0.25, 0.3)
        else:
            y = random.uniform(0.3, 0.35)
    image_ret = cv2.addWeighted(image_a, x, image_b, y, 0)
    return image_ret, cv2.addWeighted(image_a, x, image_b, y/3, 0)


#############################################################################

def translet_ran(image):
    r, c, p = image.shape
    mat = np.float64([[1, 0, random.uniform(-10, 10)],
                      [0, 1, random.uniform(-10, 10)]])
    return cv2.warpAffine(image, mat, (r, c))


def translet(image, x, y):
    r, c, p = image.shape
    mat = np.float64([[1, 0, x], [0, 1, y]])
    return cv2.warpAffine(image, mat, (r, c))


def rotate_ran(image):
    r, c, p = image.shape
    x, y = random.uniform(0, r), random.uniform(0, c)
    mat = cv2.getRotationMatrix2D(
        (x, y), random.uniform(-20, 20), random.uniform(0.9, 1.1))
    cv2.warpAffine(image, mat, (r, c))


def rotate(image, x, y, angle, rate):
    r, c, p = image.shape
    mat = cv2.getRotationMatrix2D((x, y), angle, rate)
    cv2.warpAffine(image, mat, (r, c))


def cut(image_a, image_b):  # 把两个图裁成一样大
    s = image_a.shape
    t = image_b.shape
    image_a = image_a[0:min(s[0], t[0]), 0:min(s[1], t[1])]
    image_b = image_b[0:min(s[0], t[0]), 0:min(s[1], t[1])]
    return image_a, image_b


def main():#生成二元组并输出
    ids1 = coco.getCatIds(supNms=['outdoor'])
    imgid1 = coco.getImgIds(catIds=ids1[0])

    ids2 = coco.getCatIds(supNms=['indoor'])
    imgid2 = coco.getImgIds(catIds=ids2[0])

    output=''
    for i in range(0, len(imgid1)):
        for j in range(0, len(imgid2)):
            print(i, j)
            img1 = coco.loadImgs(imgid1[i])
            # image1 = cv2.imread('%s/images/%s/%s' %
            #                    (dataDir, dataType, img1[0]['file_name']))
            img2 = coco.loadImgs(imgid2[j])
            # image2 = cv2.imread('%s/images/%s/%s' %
            #                    (dataDir, dataType, img2[0]['file_name']))
            data_path = '../../../CRRN/coco/images/val2017'
            output+=data_path+'/'+img1[0]['file_name'] + ' ' + data_path + '/' + img2[0]['file_name']+'\n'
            # out1, out2 = combine(image1, image2)
            # cv2.imwrite('E:/CCTV/CRRN/data/input{}.jpg'.format(i), out1)
            # cv2.imwrite('E:/CCTV/CRRN/data/truth{}.jpg'.ormat(i), out2)
    output_path = './train.txt'
    with open(output_path, 'w') as f:
        f.writelines(output)
    '''
    f = open('train.txt')
    for line in f:
        st = line.strip()
        le = len(st)
        st1 = st[:le // 2]
        st2 = st[le // 2 + 1:]
        img1 = cv2.imread(st1)
        img2 = cv2.imread(st2)
        img1, img2 = cut(img1, img2)
        img = combine(img1, img2)
        cv2.imshow('x', img[0])
        cv2.waitKey(0)
    '''


if __name__ == "__main__":
    main()
