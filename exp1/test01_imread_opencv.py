import cv2
from matplotlib import pyplot as plt


image_gray = cv2.imread("C:/Users/Administrator/Desktop/heangrobot.png", flags=0) # 加载灰度图像 
print(image_gray.shape)   # 图像的尺寸，按照宽度、高度显�?
print(image_gray.size)     # 图像所占内存大�?
print(image_gray.dtype)    # 夺储图像使用的数据类�?
plt.imshow(image_gray, cmap="gray")
plt.title("HEANG202210310219")
plt.show()

image_bgr = cv2.imread("C:/Users/Administrator/Desktop/heangrobot.png", flags=1) # 加载彩色（BGR）图�?
# image rgb=cv2.cvtColor(image bgr, cv2.COLOR_BGR2RGB)
image_rgb= image_bgr[:, :, ::-1]
print(image_bgr.shape)   # 高度、览度、通道�?
print(image_bgr.size)    #高度*宽度*通道�?
print(image_bgr.dtype)   # 存储图像使用的数据类�?
plt.imshow(image_rgb)
plt.title("HEANG202210310219")
plt.show()
