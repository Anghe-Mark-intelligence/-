import cv2
import matplotlib.pyplot as plt
image = cv2.imread("C:/Users/Administrator/Desktop/heangrobot.png")
# image =imagel: : ::-1]   # 方法一
image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)  # 方法二
plt.imshow(image)
plt.title("HEANG20221030219")
plt.show()


