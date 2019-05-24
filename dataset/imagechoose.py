import numpy as np
import cv2
import os

def downSample(img):
    result = cv2.resize(img, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_CUBIC)
    height, weight = result.shape[:2]
    result = result[height // 2 - 135: height // 2 + 135, weight // 2 - 135:weight // 2 + 135, :]
    return result

def bigEnough(img):
    height, weight = img.shape[:2]
    if height >= 1080 and  weight >= 1080:
        return True
    return False

def isHighFrequency(img):
    sobelx = cv2.Sobel(img, ddepth=cv2.CV_64F, dx=1, dy=0)
    sobelx = cv2.convertScaleAbs(sobelx)
    # print(np.mean(sobelx))
    sobely = cv2.Sobel(img, ddepth=cv2.CV_64F, dx=0, dy=1)
    sobely = cv2.convertScaleAbs(sobely)
    # print(np.mean(sobely))
    sobel = cv2.addWeighted(sobelx, 0.5, sobely, 0.5, 0)
    ave = np.mean(sobel)
    print("Sobel is {}".format(ave))
    if ave >= 13:
        return True
    return False

def sufficentMotion(flow):
    distance = np.sqrt(flow[:,:,0] ** 2 + flow[:,:,1] ** 2)
    # tempx = np.abs(flow[:,:,0])
    # tempx = np.array(tempx >= 8,dtype='bool')
    # tempy = np.abs(flow[:,:,1])
    # tempy = np.array(tempy >= 8,dtype='bool')
    # temp = np.logical_and(tempx,tempy)
    sum = np.sum(distance >= 8)
    shape = flow[:,:,0].size
    prop = sum /shape
    print("prop is {}".format(prop))
    if prop < 0.1:
        return False
    return True

def limitedMotion(flow):
    temp = np.abs(flow)
    mean = np.mean(temp)
    print("mean is {}".format(mean))
    if mean > 16:
        return False
    return True

def noAbruptChanges(img1,img2,flow):
    weight = int(img1.shape[1])
    height = int(img2.shape[0])
    y_coords, x_coords = np.mgrid[0:height, 0:weight]
    coords = np.float32(np.dstack([x_coords, y_coords]))
    pixel_map = coords + flow
    new_frame = cv2.remap(img1, pixel_map, None, cv2.INTER_LINEAR)
    # cv2.imshow('image', new_frame)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    l1_distance = np.linalg.norm(new_frame-img2,ord=1)
    print("l1 is {}".format(l1_distance))
    if l1_distance > 13:
        return False
    return True
def isLinearMotion(flow1,flow2):
    diff = flow1 - flow2
    absDiff = np.abs(diff)
    meanDiff = np.mean(absDiff)
    print("mean diff is {}".format(meanDiff))
    if meanDiff > 0.8:
        return False
    return True

def goodTriplet(img1,img2,img3):
    if not bigEnough(img1) or not bigEnough(img2) or not bigEnough(img3):
        print("not big enough")
        return False
    img1 = downSample(img1)
    img2 = downSample(img2)
    img3 = downSample(img3)
    if not isHighFrequency(img1) or not isHighFrequency(img2) or not isHighFrequency(img3):
        print("not high frequency")
        return False
    img1_gray = cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
    img2_gray = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)
    img3_gray = cv2.cvtColor(img3,cv2.COLOR_BGR2GRAY)
    flow1 = cv2.calcOpticalFlowFarneback(img1_gray,img2_gray,None,pyr_scale=0.5,levels=3,winsize=7,iterations=3,poly_n=5,poly_sigma=1.2,flags=cv2.OPTFLOW_FARNEBACK_GAUSSIAN)
    flow2 = cv2.calcOpticalFlowFarneback(img2_gray,img3_gray,None,pyr_scale=0.5,levels=3,winsize=7,iterations=3,poly_n=5,poly_sigma=1.2,flags=cv2.OPTFLOW_FARNEBACK_GAUSSIAN)
    if not sufficentMotion(flow1) or not sufficentMotion(flow2):
        print("not sufficent motion")
        return False
    if not limitedMotion(flow1) or not limitedMotion(flow2):
        print("not limited motion")
        return False
    if not isLinearMotion(flow1,flow2):
        print("not linear motion")
        return False
    if not noAbruptChanges(img1,img2,flow1) or not noAbruptChanges(img2,img3,flow2):
        print("not no abrupt changes")
        return False
    return True

if __name__ == '__main__':
    datafolderpath = "C://data//deeplearning//test1"
    resultpath = "c://data//deeplearning//result"
    # img1path = os.path.join(datafolderpath, "368.png")
    # img2path = os.path.join(datafolderpath, "369.png")
    # img3path = os.path.join(datafolderpath, "370.png")
    # img1 = cv2.imread(img1path)
    # img2 = cv2.imread(img2path)
    # img3 = cv2.imread(img3path)
    imglist = []
    filelist = os.listdir(datafolderpath)
    for file in filelist:
        print(file)
        imgPath = os.path.join(datafolderpath,file)
        img = cv2.imread(imgPath)
        imglist.append(img)
    length = imglist.__len__()
    for i in range(0,length-2):
        img1 = imglist[i]
        img2 = imglist[i+1]
        img3 = imglist[i+2]
        good = goodTriplet(img1,img2,img3)
        print("Triplet {0} is {1}".format(i,good))