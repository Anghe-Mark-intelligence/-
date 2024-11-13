import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['font.sans-serif'] = ['SimHei']

moon = cv.imread("C:/Users/Administrator/Desktop/heangrobot.png", 0) #灰度图
plt.imshow(moon, "gray")
plt.title("灰度图 何昂202210310219")
plt.show()
moon_f = np.copy(moon)
moon_f = moon_f.astype("float")
plt.imshow(moon_f, "gray")
plt.title("灰度图 2 何昂202210310219")
plt.show()

row, column = moon.shape
gradient = np.zeros((row, column))
for x in range(row - 1):
    for y in range(column - 1):
        gx = abs(moon_f[x + 1, y] - moon_f[x,y]) # 通过相邻像相减计算图像梯度
        gy = 	abs(moon_f[x, y + 1] - moon_f[x, y]) # 通过相邻像素相减计算图像梯度
        gradient[x, y] = gx + gy
plt.imshow(gradient, "gray")
plt.title("梯度图 何昂202210310219")
plt.show()
sharp = moon_f + gradient  # 叠加原图与梯度图，实现图像锐化
# 将小于0的像素设置为0, 将大于255 的像素设置为255
sharp = np.where(sharp < 0, 0, np.where(sharp > 255, 255, sharp))
plt.imshow(sharp, "gray")
plt.title("锐化图 何昂202210310219")
plt.show()

gradient = gradient.astype("uint8")
sharp = sharp.astype("uint8")
#显示图像
plt.subplot(1, 3, 1)
plt.imshow(moon, "gray")
plt.title("灰度图")

plt.subplot(1, 3, 2)
plt.imshow(gradient, "gray")
plt.title("梯度图 何昂202210310219")

plt.subplot(1, 3, 3)
plt.imshow(sharp, "gray")
plt.title("锐化图")
plt.show()
