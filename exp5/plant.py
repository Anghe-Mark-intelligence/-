import warnings
warnings.filterwarnings("ignore")

import os
import random
import numpy as np
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Flatten, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image

# 检查是否有可用的 GPU
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

# 设置输入图像形状
input_shape = (224, 224, 3)

# 设置训练和测试集路径
train_data_dir = 'C:/Users/Administrator/Desktop/heangcomputervision/第五次实验/PlantVillage/train'
test_data_dir = 'C:/Users/Administrator/Desktop/heangcomputervision/第五次实验/PlantVillage/test'

# 数据增强器，用于实时图像处理
train_datagen = ImageDataGenerator(
    rescale=1./255,              # 将像素值缩放到 [0,1] 之间
    shear_range=0.2,             # 随机剪切变换
    zoom_range=0.2,              # 随机缩放
    horizontal_flip=True)        # 随机水平翻转

test_datagen = ImageDataGenerator(rescale=1./255)  # 测试集不进行数据增强，仅归一化

# 创建 train_generator 和 test_generator
train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(224, 224),       # 将所有图像调整为 224x224
    batch_size=32,                # 每批次处理 32 张图像
    class_mode='categorical')     # 分类模式，适用于多分类问题

test_generator = test_datagen.flow_from_directory(
    test_data_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical')

# 1. 构建模型
def buildmodel():
    inputs = Input(shape=input_shape)
    x = Conv2D(32, (5, 5), activation='relu')(inputs)
    x = MaxPooling2D(pool_size=(3, 3))(x)
    x = Conv2D(32, (3, 3), activation='relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Conv2D(64, (3, 3), activation='relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Flatten()(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.25)(x)
    x = Dense(128, activation='relu')(x)
    outputs = Dense(train_generator.num_classes, activation='softmax')(x)

    model = Model(inputs=inputs, outputs=outputs)
    return model

# 2. 模型训练
def model_train(model):
    lr = 0.001
    epochs = 15
    opt = Adam(learning_rate=lr, decay=lr / epochs)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

    # 模型训练
    history = model.fit(
        train_generator,
        epochs=epochs,
        validation_data=test_generator,
        verbose=1
    )

    score, accuracy = model.evaluate(test_generator, verbose=1)
    print("Test score is {}".format(score))
    print("Test accuracy is {}".format(accuracy))

    # 保存模型
    export_path = f"saved_models/model_{int(time())}.h5"
    keras.models.save_model(model, export_path)
    return history, export_path

# 3. 主程序
if __name__ == '__main__':
    model = buildmodel()
    model.summary()

    # 训练模型
    history, export_path = model_train(model)

    # 打印训练曲线
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs_range = range(15)

    plt.figure(figsize=(8, 8))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.show()
