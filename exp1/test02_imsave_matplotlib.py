import numpy as np
from matplotlib import pyplot as plt
#使用Matplotlib保存uint8 类型的图像
Image_array = np.array([
[[255, 0, 0], [0, 255, 0], [0, 0, 255]],
[[255, 255, 0], [255, 0, 255], [0, 255, 255]],
[[255, 255, 255], [128, 128, 128], [0, 0, 0]]
], dtype=np.uint8)
plt.imsave("C:/Users/Administrator/Desktop/heangmagic.png", Image_array)

#读取保存的uint8类型的图像
image = plt.imread("C:/Users/Administrator/Desktop/heangmagic.png")
plt.imshow(image)
plt.title("HEANG202210310219")
plt.show()

image_array_2 = np.array([
[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
[[1, 1,0], [1, 0, 1], [0, 1, 1]],
[[1, 1, 1], [0.5, 0.5, 0.5], [0, 0, 0]]
], dtype=np.float64)
plt.imsave("C:/Users/Administrator/Desktop/heangrobot2.png", image_array_2)
plt.title("HEANG202210310219")
plt.imshow(image_array_2)
plt.show()

#读取保存的Eloat64类型的图像
image2= plt.imread("C:/Users/Administrator/Desktop/heangrobot2.png")
plt.imshow(image2)
plt.title("HEANG202210310219")
plt.show()
