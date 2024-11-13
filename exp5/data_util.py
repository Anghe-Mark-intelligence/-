import warnings
warnings.filterwarnings("ignore")

import os
import glob
import matplotlib.pyplot as plt
from time import time
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Activation, AveragePooling2D, BatchNormalization

from tensorflow.keras.preprocessing.image import ImageDataGenerator
import json
from tensorflow.keras.models import Model
import matplotlib.image as mpimg
from tensorflow.keras.optimizers import Adam

# 查看版本信息
print("Version: ", tf.__version__)
print("Eager mode:", tf.executing_eagerly())
print("GPU is", "available" if tf.config.list_physical_devices('GPU') else "NOT AVAILABLE")

### 0、定义参数
# 数据集路径
data_dir = 'C:/Users/Administrator/Desktop/heangcomputervision/第五次实验/PlantVillage'
train_dir = os.path.join(data_dir, 'train')
test_dir = os.path.join(data_dir, 'val')

# 定义数据维度和训练批次的大小
IMAGE_SHAPE = (224, 224)
input_shape = (224, 224, 3)
BATCH_SIZE = 32

# 获取类别标签
def get_files(directory):
    if not os.path.exists(directory):
        return 0
    count = 0
    for current_path, dirs, files in os.walk(directory):
        for dr in dirs:
            count += len(glob.glob(os.path.join(current_path, dr + "/*")))
    return count

# Fixing the typo 'train_dit' -> 'train_dir'
num_classes = len(glob.glob(train_dir + "/*"))

# Correcting the file path for loading the categories
with open('C:/Users/Administrator/Desktop/heangcomputervision/第五次实验/PlantVillage/categories.json', 'r') as f:
    cat_to_name = json.load(f)
    classes = list(cat_to_name.values())

print(classes)
print(len(classes))

## 1、数据预处理
# 设置数据生成器，读取源文件夹中的图像，将它们转换为‘float32”张量，并将它们(带有它们的标签)提供给网络
# 将像素值归一化在`[0, 1]范围内
# 根据选择模型对图像进行调整

test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_directory(
    test_dir,
    shuffle=False,
    seed=42,
    color_mode="rgb",
    class_mode="categorical",
    target_size=IMAGE_SHAPE,
    batch_size=BATCH_SIZE
)

# 数据增强
train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    horizontal_flip=True,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    fill_mode='nearest'
)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    shuffle=True,
    seed=42,
    color_mode="rgb",
    class_mode="categorical",
    target_size=IMAGE_SHAPE,
    batch_size=BATCH_SIZE
)
