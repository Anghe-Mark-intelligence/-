from matplotlib import pyplot as plt
image_rgb=plt.imread("C:/Users/Administrator/Desktop/heangrobot.png") # 图存储在images 文件夹下
print(image_rgb.shape)   # 高度、宽度、通道数
print(image_rgb.size)    #高度*宽度*通道数
print(image_rgb.dtype)  #存储图像使用的数据类型
plt.imshow(image_rgb)
plt.title("HEANG202210310219")
plt.show()
