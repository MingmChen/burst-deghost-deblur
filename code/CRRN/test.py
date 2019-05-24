import cv2
mat=cv2.imread("./test/output.jpg")
x,y=mat.shape[0:2]
mat=cv2.resize(mat,((int)(y*2),(int)(x*2)))
cv2.imshow("output",mat)
cv2.waitKey(0)