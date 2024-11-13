import os
import shutil
import random
from sklearn.model_selection import train_test_split

# 定义数据集路径
data_dir = r'C:\Users\Administrator\Desktop\heangcomputervision\第五次实验\PlantVillage\Plant_leave_diseases_dataset_with_augmentation'
train_dir = os.path.join(data_dir, 'train')
val_dir = os.path.join(data_dir, 'val')

# 创建 train 和 val 文件夹
if not os.path.exists(train_dir):
    os.makedirs(train_dir)
if not os.path.exists(val_dir):
    os.makedirs(val_dir)

# 定义划分比例
train_ratio = 0.8  # 80% 训练集
val_ratio = 0.2    # 20% 验证集

# 获取所有类别文件夹
categories = os.listdir(data_dir)

# 排除 train 和 val 文件夹，以免重复处理
categories = [cat for cat in categories if cat not in ['train', 'val']]

# 遍历每个类别文件夹
for category in categories:
    category_path = os.path.join(data_dir, category)
    images = os.listdir(category_path)
    
    # 将图片随机划分为训练集和验证集
    train_images, val_images = train_test_split(images, test_size=val_ratio, random_state=42)

    # 为当前类别创建对应的 train 和 val 文件夹
    train_category_dir = os.path.join(train_dir, category)
    val_category_dir = os.path.join(val_dir, category)
    
    if not os.path.exists(train_category_dir):
        os.makedirs(train_category_dir)
    if not os.path.exists(val_category_dir):
        os.makedirs(val_category_dir)
    
    # 复制图片到训练集文件夹
    for image in train_images:
        src_image_path = os.path.join(category_path, image)
        dest_image_path = os.path.join(train_category_dir, image)
        shutil.copyfile(src_image_path, dest_image_path)
    
    # 复制图片到验证集文件夹
    for image in val_images:
        src_image_path = os.path.join(category_path, image)
        dest_image_path = os.path.join(val_category_dir, image)
        shutil.copyfile(src_image_path, dest_image_path)

print("数据划分完成，训练集和验证集已生成。")
