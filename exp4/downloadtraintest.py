import os
import numpy as np
from PIL import Image
from tensorflow.keras.datasets import mnist

# 加载 MNIST 数据集
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
#何昂magic1
# 缩放图像至 16x16
def resize_and_save_images(images, labels, folder):
    if not os.path.exists(folder):
        os.makedirs(folder)
    for i, (img, label) in enumerate(zip(images, labels)):
        # 缩放至 16x16 像素
        img_resized = Image.fromarray(img).resize((16, 16), Image.ANTIALIAS)
        # 保存为 jpg 文件，文件名格式为: label_index.jpg
        img_resized.save(os.path.join(folder, f"{label}_{i}.jpg"))

# 保存训练集和测试集的图像
train_folder = r"C:\Users\Administrator\Desktop\heangcomputervision\第四次实验\img_train"
test_folder = r"C:\Users\Administrator\Desktop\heangcomputervision\第四次实验\img_test"

# 分别保存 100 张为测试集，剩下的为训练集
resize_and_save_images(train_images[100:], train_labels[100:], train_folder)
resize_and_save_images(train_images[:100], train_labels[:100], test_folder)
