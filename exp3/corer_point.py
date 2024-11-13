import cv2
import numpy as np

#读取图像
img=cv2.imread("C:/Users/Administrator/Desktop/heangrobot.png")
#将图像转化成灰度图
gray=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray=np.float32(gray)

#gray: 输入的float类型的灰度图
#2:检测过程中考虑的领域大小
#3:使用Scbel算法在求导时使用的窗口大小
#0.04:Harris角点检测方程中的自由参数，取值范围为[0.04,  0.06]
dst= cv2.cornerHarris(gray, 2, 3, 0.04)
#这里设定一个阈值，只要大于等于这个阈值就可认判定为角点
img[dst>0.01*dst.max()]=[0, 0, 255] #[0, 0, 255]为红色

cv2.imshow('dst何昂202210310219' ,img)
if cv2.waitKey(0) & 0xff==27:
    cv2.destroyAllWindows()
