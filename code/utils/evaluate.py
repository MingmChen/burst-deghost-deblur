#coding:utf-8
import sys
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt

data_dir =  "./"

old_path  = data_dir + "1.jpeg"
new_path  = data_dir + "2.jpeg"

gap = [0,1]

feature_params = dict( maxCorners = 100000,
                       qualityLevel = 0.001,
                       minDistance = 7,
                       blockSize = 7 )

lk_params = dict( winSize = (15,15), 
                  maxLevel = 3,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))


def cal_lk(old_path, new_path):
    old_img = cv2.imread(old_path)
    old_gray = cv2.cvtColor(old_img, cv2.COLOR_BGR2GRAY)
    new_img = cv2.imread(new_path)
    new_gray = cv2.cvtColor(new_img, cv2.COLOR_BGR2GRAY)
    
    color = np.random.randint(0,255,(100000,3))
    mask = np.zeros_like(old_img)
    
    p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)
    p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, new_gray, p0, None, **lk_params)

    # select good points
    good_new = p1[st == 1]
    good_old = p0[st == 1]
    good = good_new - good_old

    for i, (new, old) in enumerate(zip(good_new, good_old)):
        a, b = new.ravel()
        c, d = old.ravel()
        gapx = abs(a-c)
        gapy = abs(b-d)
        if gapx >= gap[0] and gapy >= gap[0] and gapx <= gap[1] and gapy <=gap[1]:
            mask = cv2.line(mask, (a,b), (c,d), color[i].tolist(), 2)
            frame = cv2.circle(new_img, (a,b), 5, color[i].tolist(),-1)
    img = cv2.add(frame, mask)

    # plot
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.subplot(221)
    plt.imshow(img)
    plt.subplot(222)
    plt.imshow(mask,  cmap=plt.cm.gray)
    plt.subplot(223)

    n1, bins1, patches1 = plt.hist(good[:,0], bins=200, normed=0, facecolor='green', alpha=0.75)
    plt.subplot(224)
    n2, bins2, patches2 = plt.hist(good[:,1], bins=200, normed=0, facecolor='green', alpha=0.75)
    ind1 = np.arange(len(bins1)-1)
    ind2 = np.arange(len(bins2)-1)
    width = 0.35
    fig, ax = plt.subplots()

    rects1 = ax.bar(bins1[0:-1], n1, width, label = "x displacement",lw=1, alpha=0.3)
    rects2 = ax.bar(bins2[0:-1]+width, n2, width, label = "y displacement", lw=1, alpha=0.3)
    plt.xticks(np.arange(-30,30,1))
    plt.legend(loc="upper left")

    plt.show()


if __name__ == '__main__':

    if len(sys.argv) == 3:
        old_path = data_dir + sys.argv[1]
        new_path = data_dir + sys.argv[2]
        cal_lk(old_path, new_path)

    elif len(sys.argv) == 1:
        cal_lk(old_path, new_path)

    else:
        raise ValueError

