# 步骤一: 导入模块
import cv2

# 步骤二: 读取图像，并将其转换为灰度图
img = cv2.imread("C:/Users/Administrator/Desktop/heangrobot.png")
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 步骤三: 采用二值化方式处理图像
# 像素值在182和255之间的数据为255（白色），小于182的数据为0（黑色）
ret, thresh = cv2.threshold(img_gray, 182, 255, cv2.THRESH_BINARY)

# 步骤四: 查找轮廓
# 使用简易方式提取全部轮廓
contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# 步骤五: 绘制轮廓
# 传入的参数: 图像、轮廓坐标、全部轮廓、轮廓颜色(红色), 线宽
img_contour = cv2.drawContours(img.copy(), contours, -1, (0, 0, 255), 2)
# 步骤六: 可视化图像
cv2.imshow('gray何昂202210310219', img_gray)  # 灰度图效果
cv2.imshow('bin何昂202210310219', thresh)  # 二值化图效果
cv2.imshow('contour何昂202210310219', img_contour)  # 轮廓图效果
# 按任意键退出图像显示，结束程序
cv2.waitKey(0)
cv2.destroyAllWindows()
