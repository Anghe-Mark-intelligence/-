import cv2
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['font.sans-serif']=['SimHei'] #指定默认字体为黑体
mpl.rcParams['axes.unicode_minus'] = False #正常显示负号
img = cv2.imread("C:/Users/Administrator/Desktop/heangrobot.png", 0) # 直接读为灰度图
blur = cv2.blur(img, (5, 5))# 模板大小为5*5
#显示图像
plt.subplot(1, 2, 1)
plt.imshow(img, 'gray')
plt.title('zaosheng 何昂202210310219')
plt.subplot(1, 2, 2)
plt.imshow(blur, 'gray')
plt.title('blur')
plt.show()

img = cv2.imread("C:/Users/Administrator/Desktop/heangrobot.png", 0)# 直接读为灰度图
dst = cv2.medianBlur(img, (5))# 卷积核大小为5
#显示图像
plt.subplot(1, 2, 1)
plt.imshow(img, 'gray')
plt.title('zaosheng 何昂202210310219')
plt.subplot(1, 2, 2)
plt.imshow(dst, 'gray')
plt.title("medianBlur")
plt.show()

#高斯滤波
img =cv2.imread("C:/Users/Administrator/Desktop/heangrobot.png", 0)# 直接读为灰度图像
m_dst = cv2.medianBlur(img,  (5))
g_dst = cv2.GaussianBlur(img, (5, 5), 0)# 高斯核为5*5
#显示图像
plt.subplot(1, 3, 1), plt.imshow(img, 'gray')
plt.title("zaosheng")
plt.subplot(1, 3, 2), plt.imshow(g_dst, 'gray')
plt.title('GaussianBlurn 何昂202210310219')
plt.subplot(1, 3, 3), plt.imshow(m_dst, 'gray')
plt.title('mediaBlur')
plt.show()
