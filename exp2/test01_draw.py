import cv2
import numpy as np
import matplotlib.pyplot as plt

#1.绘制线
img = np.zeros((512, 512, 3), np.uint8)
print(img.dtype)
cv2.line(img, # 目标图像
(0, 0),  # 起点
(256, 256) , # 终点
(255, 0, 0), #颜色
5) #粗细
img_line = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.imshow(img_line)
plt.title('HEANG202210310219')
plt.show()

# 2.绘制矩形
img = np.zeros((512, 512, 3), np.uint8)
cv2.rectangle(img, #目标图像
(128, 128), #顶点
(256, 256), #相对的顶点
(0, 255, 0), #颜色
3)	#粗细
img_rectangle = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.imshow(img_rectangle)
plt.title("HEANG202210310219")
plt.show()

# 3.绘制圆形
img = np.zeros((512, 512, 3), np.uint8)
cv2.circle(img, #目标图像
(256, 256),  # 圆心
256, #半径
(0, 0, 255),  #颜色
-1)  #填充
img2 = cv2.cvtColor(img,  cv2.COLOR_BGR2RGB)
plt.imshow(img2)
plt.title("HEANG202210310219")
plt.show()

# 4.绘制椭圆形
img = np.zeros((512, 512, 3), np.uint8)
cv2.ellipse(img,  #目标图像
(256, 256), # 中心
(256, 128), # 长轴、短轴
0, # 逆时针旋转角度
0, #开始角度
360, # 结束角度
(0, 0, 255), #颜色
-1) #填充
cv2.ellipse(img, (256, 256), (256, 128), 45, 0, 360, (0, 255, 0), -1)
cv2.ellipse(img, (256, 256), (256, 128), 90, 0, 360, (255, 0, 0), -1)

img2 = cv2.cvtColor(img,  cv2.COLOR_BGR2RGB)
plt.imshow(img2)
plt.title("HEANG202210310219")
plt.show()

# 5.绘制多边形
img = np.zeros((512, 512, 3), np.uint8)
pts = np.array([[50, 50], [400,100], [462,462], [100,400]],np.int64)
print(pts)
print(pts.shape)
pts = pts.reshape((-1, 1, 2))
print(pts)
print(pts.shape)
cv2.polylines(img, #目标图像
[pts], # 顶点
True, # 是否闭合
(0, 0, 255) ,#颜色
3)  #粗细
img2 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.imshow(img2)
plt.title("HEANG202210310219")
plt.show()

#6.添加文字
img =np.zeros((512, 512, 3), np.uint8)
font = cv2.FONT_HERSHEY_SIMPLEX
cv2.putText(img, # 目标图像
"OpenCV", #文字
(10, 300), # 文本框左下角
font, #文宇字体
4,# 文字大小
(255, 255, 255), # 文宇颜色
3,#文宇粗细
cv2.LINE_AA #文字线型
)
img2 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.imshow(img2)
plt.title("HEANG202210310219")
plt.show()
