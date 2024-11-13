import cv2
import matplotlib.pyplot as plt
# 绘图文字使用黑体显示(显示中文，默认不支持中文)
plt.rcParams['font.sans-serif'] = ['SimHei']
img=cv2.imread("C:/Users/Administrator/Desktop/heangrobot.png")
#将彩色图转换为灰度图
img= cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
edges1 = cv2.Canny(img, 200, 300)
plt.subplot(121)  # 会制第一张子图，总共为1行2列
plt.title('原图 何昂202210310219')
plt.imshow(img)
#去除图像的坐标尺
plt.xticks([])
plt.yticks([])
plt.subplot(122) # 给制第二张子图，总共为2行2列
plt.title('轮廓处理 1 何昂202210310219')
plt.imshow(edges1, cmap='gray')
plt.xticks([])
plt.yticks([])
plt.show()