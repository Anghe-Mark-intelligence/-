#导入模块
import cv2
from matplotlib import pyplot as plt
#读取图像
img = cv2.imread("C:/Users/Administrator/Desktop/heangrobot.png")
print(img.shape)
width, height = img.shape[:2]
#显示图像
img_original = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.imshow(img_original)
plt.title('HEANG202210310219')
plt.show()

# 1.通过 dsize 设置输出图像的大小
img_dsize = cv2.resize(img,  # 输入图像
(4 *width, 2*height), # 输出图像的大小
)
img_dsize = cv2.cvtColor(img_dsize, cv2.COLOR_BGR2RGB)
plt.imshow(img_dsize)
plt.title('HEANG202210310219')
plt.show()

img_dsize = cv2.cvtColor(img_dsize, cv2.COLOR_BGR2RGB)
plt.imshow(img_dsize)
plt.title('HEANG202210310219')
plt.show()

#2.通过 fx和fy 设置输出图像的大小
img_fx_fy = cv2.resize(img,# 输入图像
None ,  # 输出图像的大小
fx=1/2,  #y轴缩放因子
fy=1/4,  #x轴缩放因子
)
img_fx_fy = cv2.cvtColor(img_fx_fy, cv2.COLOR_BGR2RGB)
plt.imshow(img_fx_fy)
plt.show()
