# 导入必要的模块
import os
import sys
import tkinter as tk
from tkinter import filedialog
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import skimage.io
import tensorflow as tf

# 如果使用的是 TensorFlow 2.x，请取消注释以下行
# tf.compat.v1.disable_eager_execution()

# 设置 Matplotlib 的后端，以便在 Tkinter 中显示图像
matplotlib.use('TkAgg')

# 设置项目根目录
ROOT_DIR = os.path.abspath("./")

# 导入 Mask RCNN 模块
sys.path.append(ROOT_DIR)
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize

# 导入配置
from mrcnn.config import Config

# 定义模型配置
class InferenceConfig(Config):
    NAME = "tomato"
    IMAGES_PER_GPU = 1
    NUM_CLASSES = 1 + 1  # 背景 + 番茄
    GPU_COUNT = 1
    DETECTION_MIN_CONFIDENCE = 0.7
    IMAGE_MIN_DIM = 800
    IMAGE_MAX_DIM = 1024
    # 如果需要其他配置，请在此添加

inference_config = InferenceConfig()

# 模型文件的路径
MODEL_DIR = r"C:\Users\Administrator\Desktop\heangcomputervision\shiyan6\model\20241108T225218_1"
MODEL_PATH = os.path.join(MODEL_DIR, "mask_rcnn_coco.h5")

# 检查模型文件是否存在
if not os.path.exists(MODEL_PATH):
    print(f"模型文件未找到：{MODEL_PATH}")
    sys.exit(1)

# 创建模型对象并加载权重
model = modellib.MaskRCNN(mode="inference", config=inference_config, model_dir=MODEL_DIR)

# 只替换非法字符，不添加索引
for layer in model.keras_model.layers:
    clean_name = layer.name.replace('/', '_').replace(':', '_')
    layer._name = clean_name

print("从以下路径加载模型权重：", MODEL_PATH)
model.load_weights(MODEL_PATH, by_name=True)

# 类别名称列表
class_names = ['BG', 'tomato']  # 背景和番茄

# 定义用于选择图像并进行检测的函数
def detect_and_display():
    # 打开文件选择对话框
    file_path = filedialog.askopenfilename(
        title='选择图像文件',
        filetypes=[('Image Files', '*.jpg;*.jpeg;*.png')]
    )
    if not file_path:
        print("未选择任何文件。")
        return

    print("选择的图像文件：", file_path)

    # 读取图像
    image = skimage.io.imread(file_path)

    # 检测
    results = model.detect([image], verbose=1)
    r = results[0]

    # 可视化结果
    plt.figure(figsize=(12, 12))
    visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'],
                                class_names, r['scores'], ax=plt.gca())
    plt.show()

# 创建主窗口
root = tk.Tk()
root.title("番茄检测与分割")

# 创建选择图像并检测的按钮
btn = tk.Button(root, text='选择图像并检测', command=detect_and_display)
btn.pack(pady=20)

# 运行主循环
root.mainloop()
