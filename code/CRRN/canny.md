# 关于canny参数设置

### 原图：

![1550155008529](C:\Users\dell\AppData\Roaming\Typora\typora-user-images\1550155008529.png)

##1、有无高斯滤波

### 无：

![1550154974996](C:\Users\dell\AppData\Roaming\Typora\typora-user-images\1550154974996.png)

### 有：image = cv2.GaussianBlur(image, (3, 3), 0)

![1550155048796](C:\Users\dell\AppData\Roaming\Typora\typora-user-images\1550155048796.png)

可以发现少了很多不重要的轮廓，画面没有那么杂乱了

##2、高斯滤波参数设置：

### (5,5),0:

![1550155289646](C:\Users\dell\AppData\Roaming\Typora\typora-user-images\1550155289646.png)

### (1,1),0:

![1550155341954](C:\Users\dell\AppData\Roaming\Typora\typora-user-images\1550155341954.png)

可以发现(1,1)时和没有高斯滤波没有差别，(5,5)时画面更加简洁，综合来看，(3,3)比较适中

**sigmaX和sigmaY都为0时由ksize计算得来**

## 3、canny参数设置

关于2个阈值参数：

1. 低于阈值1的像素点会被认为不是边缘；

2. 高于阈值2的像素点会被认为是边缘；

3. 在阈值1和阈值2之间的像素点,若与第2步得到的边缘像素点相邻，则被认为是边缘，否则被认为不是边缘。

   ### (10,150):

   ![1550155781242](C:\Users\dell\AppData\Roaming\Typora\typora-user-images\1550155781242.png)

   多了很多莫名其妙的边缘

   ### (50,500):

   ![1550155822288](C:\Users\dell\AppData\Roaming\Typora\typora-user-images\1550155822288.png)

   唔。。。惨不忍睹，脸都没了
