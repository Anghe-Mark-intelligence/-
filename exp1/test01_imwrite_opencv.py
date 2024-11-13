import numpy as np
import cv2
from matplotlib import pyplot as plt
#使用OpenCV保存uint8类型的图像
Image_array =np.array([
[[255, 0, 0], [0, 255, 0], [0, 0, 255]],
 [[255, 255, 0], [255, 0, 255], [0, 255, 255]],
[[255, 255, 255], [128, 128, 128], [0, 0, 0]]
], dtype=np.uint8)
cv2.imwrite("C:/Users/Administrator/Desktop/heangmagic.png",  Image_array)

#读取保存的uint8类型的图像
image = cv2.imread("C:/Users/Administrator/Desktop/heangmagic.png")
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
plt.imshow(image)
plt.title("HEANG202210310219")
plt.show()
#使用OpenCV保存float64 类型的图像

Image_array_2 = np.array([
[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
[[1, 1, 0], [1, 0, 1], [0, 1, 1]],
[[1, 1, 1], [0.5, 0.5, 0.5], [0, 0,0]]
], dtype=np.float64)
cv2.imwrite("C:/Users/Administrator/Desktop/heangmagic.png", Image_array_2)
# 读取保存的float64 类型的图像

image = cv2.imread("C:/Users/Administrator/Desktop/heangmagic.png")
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
plt.imshow(image)
plt.title("HEANG202210310219")
plt.show()

# 使用OpenCV保存由float64 类型转换为uint8 类型的图像
image_array_2_cvt = Image_array_2 * 255
image_array_2_cvt = image_array_2_cvt.astype(np.uint8)
cv2.imwrite("C:/Users/Administrator/Desktop/heangmagic.png", Image_array)

# 读取保存的由foat64类型转换为uint8类型的图像
image= cv2.imread("C:/Users/Administrator/Desktop/heangmagic.png")
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
plt.imshow(image)
plt.title("HEANG202210310219")
plt.show()
