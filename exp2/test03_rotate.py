#导入模块
import cv2
from matplotlib import pyplot as plt
#读取图像
img = cv2.imread("C:/Users/Administrator/Desktop/heangrobot.png")
print(img.shape)
Width, height = img.shape[:2]
# 显示旋转前的图像
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.imshow(img_rgb)
plt.title('heang202210310219before rotate')
plt.show()
#2*3变换矩阵
M = cv2.getRotationMatrix2D((Width/2, height/2),  # 旋转中心
45,		# 旋转角度
1)	    # 缩放比例
print(M)
#旋转
img_rotate = cv2.warpAffine(img,	# 输入图像
M,     # 变换矩阵
(Width, height))   # 变换后的图像大小
img_rotate = cv2.cvtColor(img_rotate, cv2.COLOR_BGR2RGB)
plt.imshow(img_rotate)
plt.title('heang202210310219after rotate')
plt.show()
