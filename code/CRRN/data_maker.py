from pycocotools.coco import COCO
import cv2
import random
import numpy as np
import sys
sys.path.append(
    'C:/Users/dell/Desktop/study_3/CCTV/CRRN/coco/windows/cocoapi-master/PythonAPI')


def main():
    dataDir = 'E:/CCTV/CRRN/coco'
    dataType = 'val2017'
    annFile = '{}/annotations/instances_{}.json'.format(dataDir, dataType)
    coco = COCO(annFile)

    def combine(image_a, image_b):
        x = random.uniform(0.8, 1)
        y = random.uniform(0.1, 0.5)
        image_ret = cv2.addWeighted(image_a, x, image_b, y, 0)
        return image_ret, image_a

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

    def cut(image_a, image_b):
        s = image_a.shape
        t = image_b.shape
        image_a = image_a[0:min(s[0], t[0]), 0:min(s[1], t[1])]
        image_b = image_b[0:min(s[0], t[0]), 0:min(s[1], t[1])]
        return image_a, image_b

    ids1 = coco.getCatIds(catNms=['indoor'])
    imgid1 = coco.getImgIds(catIds=ids1)

    ids2 = coco.getCatIds(catNms=['person'])
    imgid2 = coco.getImgIds(catIds=ids2)

    for i in range(0, 100):
        img1 = coco.loadImgs(imgid1[i])
        image1 = cv2.imread('%s/images/%s/%s' %
                            (dataDir, dataType, img1[0]['file_name']))

        img2 = coco.loadImgs(imgid2[i])
        image2 = cv2.imread('%s/images/%s/%s' %
                            (dataDir, dataType, img2[0]['file_name']))

        image1, image2 = cut(image1, image2)
        out1, out2 = combine(image1, image2)

        cv2.imwrite('E:/CCTV/CRRN/data/input{}.jpg'.format(i), out1)
        cv2.imwrite('E:/CCTV/CRRN/data/turth{}.jpg'.format(i), out1)


if __name__ == "__main__":
    main()
