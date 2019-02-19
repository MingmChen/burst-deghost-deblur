import math
import numpy as np
import cv2

def fspecial(len,theta,size):

	alpha=math.floor(theta)/180*math.pi
	print(alpha)
	cosa=math.cos(alpha)
	sina=math.sin(alpha)
	Hmat=np.zeros((size+1,size+1))
	half=int(size/2)

	p1x=len/2.0*cosa
	p1y=len/2.0*sina
	p2x=-len/2.0*cosa
	p2y=-len/2.0*sina

	eps=1e-6

	#计算必要参数
	for x in range(-half,size-half+1):
		for y in range(-half,size-half+1):
			if(math.fabs(cosa*x+sina*y)>len/2.0):	#(x,y).*(cosa,sina),投影的长度
				if(math.fabs(sina*x-cosa*y)<eps):	#在直线上
					v=0
				else:
					h=math.sqrt(min((x-p1x)*(x-p1x)+(y-p1y)*(y-p1y),(x-p2x)*(x-p2x)+(y-p2y)*(y-p2y)))	#到端点的距离
					v=max(1-h,0)
			else:
				h=math.fabs(sina*x-cosa*y)	#点到直线距离
				v=max(1-h,0)
			j=x+half
			i=size-(y+half)
			Hmat[i][j]=v
			print(i,j,v)
	Hmat=Hmat/Hmat.sum()
	return Hmat

a=fspecial(10,45,10)
