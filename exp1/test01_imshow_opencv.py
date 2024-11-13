import cv2

image = cv2.imread("C:/Users/Administrator/Desktop/heangrobot.png")# 读取图像

cv2.namedWindow("何昂202210310219", cv2.WINDOW_NORMAL)# 创建一个窗口
cv2.resizeWindow("何昂202210310219", 300, 200)  # 设置窗口大小

cv2.imshow("何昂202210310219", image)# 在窗口中显示图像

cv2.waitKey(0)# 等待按键输入

cv2.destroyAllWindows()# 销毁所有窗口
