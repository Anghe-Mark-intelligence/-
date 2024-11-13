#导入模块
import cv2
import numpy as np
from matplotlib import pyplot as plt

# 读取图像
img = cv2.imread("C:/Users/Administrator/Desktop/heangrobot.png")
print(img.shape)
width,height = img.shape[:2]

# 显示平移前的图像
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # 颜色通道从BGR转换为RGB
plt.imshow(img_rgb)
plt.title('heang202210310219')
plt.show()
# 2*3 变换矩阵:100表示水平方向上的平移距离，50表示垂直方向上的平移距离
M = np.float64([[1, 0, 100], [0, 1,50]])
#平移
img2 = cv2.warpAffine(img,  # 变换前的图像
M,#变换矩阵
(width, height))  #变换后的图像大小
img2_rgb = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)# 颜色通道从BGR 转换为RGB
plt.imshow(img2_rgb)
plt.title('heang202210310219')
plt.show()
