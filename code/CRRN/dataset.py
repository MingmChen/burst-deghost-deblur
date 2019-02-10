import cv2
import numpy as np
dir_data='../../../CRRN/data/'

def gradient(image):
    gx=cv2.Sobel(image,cv2.CV_64F,1,0)
    gy=cv2.Sobel(image,cv2.CV_64F,0,1)
    gxx=cv2.convertScaleAbs(gx)
    gyy=cv2.convertScaleAbs(gy)
    gxy=cv2.addWeighted(gxx,0.5,gyy,0.5,0)
    return gxy

def mix(x):
    return cv2.imread(dir_data+'input{}.jpg'.format(x))

def tru(x):
    return cv2.imread(dir_data+'truth{}.jpg'.format(x))

def GT(x):
    image=cv2.imread(dir_data+'truth{}.jpg'.format(x))
    return cv2.cvtColor(gradient(image),cv2.COLOR_RGB2GRAY)

data_h=224
data_w=288

def unit_inputs(x,batch_size):
    ret=np.empty([batch_size,4,data_h,data_w])
    for i in range(0,4):
        image_mix=mix(i+x)
        image_mix_gra=cv2.cvtColor(gradient(image_mix),cv2.COLOR_RGB2GRAY)
        image_mix=cv2.resize(image_mix,dsize=(data_w,data_h),fx=1,fy=1)
        image_mix_gra=cv2.resize(image_mix_gra,dsize=(data_w,data_h),fx=1,fy=1)
        cv2.imshow('a',image_mix_gra)
        cv2.waitKey(0)
        for ii in range(0,data_h):
            for j in range(0,data_w):
                for k in range(0,3):
                    ret[i][k][ii][j]=image_mix[ii][j][k]
                ret[i][3][ii][j]=image_mix_gra[ii][j]
    return ret

unit_inputs(0,4)