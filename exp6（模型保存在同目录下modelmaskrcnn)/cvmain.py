#conda tf1_env
#### 1.1 导入模块
import os
import sys
import random
import math
import numpy as np
import cv2
import json
import datetime
import matplotlib
matplotlib.use('Agg')  # 如果在没有 GUI 的环境中运行，可以使用 'Agg' 后端
import matplotlib.pyplot as plt
from collections import OrderedDict
from skimage.draw import polygon
import skimage

# 设置项目根目录
ROOT_DIR = os.path.abspath("./")
print(os.listdir(ROOT_DIR))

# 导入 Mask RCNN 模块
sys.path.append(ROOT_DIR)
from mrcnn.config import Config
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize

# 导入 TensorFlow 和 Keras
import tensorflow as tf
from tensorflow.keras.backend import clear_session
clear_session()

# 禁用急切执行模式（如果使用 TensorFlow 2.x，请取消下一行注释）
# tf.compat.v1.disable_eager_execution()

#### 1.2 配置模型结构与训练参数
class TomatoConfig(Config):
    NAME = "tomato"
    IMAGES_PER_GPU = 1
    NUM_CLASSES = 1 + 1  # 背景 + 番茄
    STEPS_PER_EPOCH = 100  # 根据您的数据集大小调整
    VALIDATION_STEPS = 10  # 根据您的数据集大小调整
    DETECTION_MIN_CONFIDENCE = 0.7  # 降低置信度阈值
    IMAGE_MIN_DIM = 800
    IMAGE_MAX_DIM = 1024
    BACKBONE_STRIDES = [4, 8, 16, 32, 64]

    # 动态计算 BACKBONE_SHAPES
    def __init__(self):
        super().__init__()
        self.BACKBONE_SHAPES = self.compute_backbone_shapes()

    def compute_backbone_shapes(self):
        return np.array(
            [[int(math.ceil(self.IMAGE_SHAPE[0] / stride)),
              int(math.ceil(self.IMAGE_SHAPE[1] / stride))] for stride in self.BACKBONE_STRIDES]
        )

config = TomatoConfig()
config.display()

#### 可视化窗口
def get_ax(rows=1, cols=1, size=8):
    _, ax = plt.subplots(rows, cols, figsize=(size*cols, size*rows))
    return ax

#### 数据预处理
class TomatoDataset(utils.Dataset):
    def load_tomato(self, dataset_dir, subset):
        self.add_class("tomato", 1, "tomato")
        assert subset in ["Train", "Test"]
        
        img_dir = os.path.join(dataset_dir, "img")
        ann_dir = os.path.join(dataset_dir, "ann")

        for filename in os.listdir(ann_dir):
            if filename.endswith(".json"):
                annotation_path = os.path.join(ann_dir, filename)
                with open(annotation_path) as f:
                    annotation = json.load(f)

                image_filename = filename.replace(".jpg.json", ".jpg")
                image_path = os.path.join(img_dir, image_filename)
                
                if not os.path.exists(image_path):
                    print(f"Warning: Image file {image_path} does not exist.")
                    continue

                if 'objects' in annotation and annotation['objects']:
                    polygons = [obj['points']['exterior'] for obj in annotation['objects']]
                    image = skimage.io.imread(image_path)
                    height, width = image.shape[:2]

                    self.add_image(
                        "tomato",
                        image_id=image_filename,
                        path=image_path,
                        width=width, height=height,
                        polygons=polygons
                    )
                    print(f"Loaded {image_filename} with {len(polygons)} polygons.")

    def load_mask(self, image_id):
        image_info = self.image_info[image_id]
        if image_info["source"] != "tomato":
            return super(self.__class__, self).load_mask(image_id)

        info = self.image_info[image_id]
        mask = np.zeros([info["height"], info["width"], len(info["polygons"])], dtype=np.uint8)
        for i, p in enumerate(info["polygons"]):
            rr, cc = polygon([point[1] for point in p], [point[0] for point in p], mask.shape[:2])
            mask[rr, cc, i] = 1
        return mask.astype(bool), np.ones([mask.shape[-1]], dtype=np.int32)

    def image_reference(self, image_id):
        info = self.image_info[image_id]
        if info["source"] == "tomato":
            return info["path"]
        else:
            return super(self.__class__, self).image_reference(image_id)

# 定义训练和测试数据集路径
train_dataset_dir = r'C:\Users\Administrator\Desktop\heangcomputervision\shiyan6\laboro_tomato_DatasetNinja\Train'
test_dataset_dir = r'C:\Users\Administrator\Desktop\heangcomputervision\shiyan6\laboro_tomato_DatasetNinja\Test'

dataset_train = TomatoDataset()
dataset_train.load_tomato(train_dataset_dir, "Train")
dataset_train.prepare()

dataset_val = TomatoDataset()
dataset_val.load_tomato(test_dataset_dir, "Test")
dataset_val.prepare()

# 验证数据集是否加载成功
if len(dataset_train.image_ids) == 0:
    print("No images found in the training dataset.")
if len(dataset_val.image_ids) == 0:
    print("No images found in the validation dataset.")

# 再次清理会话，确保一致性
clear_session()

# 指定唯一的模型目录
MODEL_ROOT_DIR = r"C:\Users\Administrator\Desktop\heangcomputervision\modelmaskrcnn"
timestamp = datetime.datetime.now().strftime("%Y%m%dT%H%M%S")
MODEL_DIR = os.path.join(MODEL_ROOT_DIR, timestamp)
if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)
print(f"Model directory created at: {MODEL_DIR}")

# COCO 预训练权重的路径
COCO_MODEL_PATH = os.path.join(MODEL_ROOT_DIR, "mask_rcnn_coco.h5")

# 如果需要，则下载 COCO 预训练模型
if not os.path.exists(COCO_MODEL_PATH):
    print(f"Downloading pretrained model to {COCO_MODEL_PATH} ...")
    utils.download_trained_weights(COCO_MODEL_PATH)
    print("... done downloading pretrained model!")

# 初始化模型
model = modellib.MaskRCNN(mode="training", config=config, model_dir=MODEL_DIR)

# 只替换非法字符，不添加索引
for layer in model.keras_model.layers:
    clean_name = layer.name.replace('/', '_').replace(':', '_')
    layer._name = clean_name

# 打印模型中所有层的名称进行验证
for layer in model.keras_model.layers:
    print(layer.name)

# 选择初始化方式
init_with = "coco"

if init_with == "coco":
    model.load_weights(COCO_MODEL_PATH, by_name=True, exclude=[
        "mrcnn_class_logits", "mrcnn_bbox_fc", "mrcnn_bbox", "mrcnn_mask"])
elif init_with == "last":
    model.load_weights(model.find_last(), by_name=True)

# 开始训练，首先训练头部
model.train(dataset_train, dataset_val,
            learning_rate=config.LEARNING_RATE,
            epochs=10,  # 增加训练轮数
            layers='heads')

# 继续训练所有层
model.train(dataset_train, dataset_val,
            learning_rate=config.LEARNING_RATE / 10,
            epochs=20,  # 增加训练轮数
            layers="all")

# 只保存模型的权重，不保存整个模型
model_path = os.path.join(MODEL_DIR, "mask_rcnn_tomato.h5")
model.keras_model.save_weights(model_path)
print(f"Model weights saved at: {model_path}")

#### 步骤四：测试

class InferenceConfig(TomatoConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    USE_MINI_MASK = False

inference_config = InferenceConfig()

clear_session()
model = modellib.MaskRCNN(mode="inference", config=inference_config, model_dir=MODEL_DIR)

# 加载训练好的权重
print("Loading weights from ", model_path)
model.load_weights(model_path, by_name=True)

# 随机选择一个验证图像进行测试
image_id = random.choice(dataset_val.image_ids)
original_image, image_meta, gt_class_id, gt_bbox, gt_mask = modellib.load_image_gt(
    dataset_val, inference_config, image_id)

# 可视化真实的掩码
visualize.display_instances(original_image, gt_bbox, gt_mask, gt_class_id, 
                            dataset_val.class_names, figsize=(8, 8))

results = model.detect([original_image], verbose=1)

r = results[0]
visualize.display_instances(original_image, r['rois'], r['masks'], r['class_ids'], 
                            dataset_val.class_names, r['scores'], ax=get_ax())

# 在验证集的一个子集上评估 mAP
image_ids = np.random.choice(dataset_val.image_ids, 10)
APs = []
for image_id in image_ids:
    image, image_meta, gt_class_id, gt_bbox, gt_mask = modellib.load_image_gt(
        dataset_val, inference_config, image_id)
    molded_images = np.expand_dims(modellib.mold_image(image, inference_config), 0)

    results = model.detect([image], verbose=0)
    r = results[0]

    # 如果没有检测到任何对象，跳过该图像
    if len(r["class_ids"]) == 0:
        print(f"No detections for image {image_id}, skipping.")
        APs.append(0)
        continue

    AP, precisions, recalls, overlaps = utils.compute_ap(
        gt_bbox, gt_class_id, gt_mask,
        r["rois"], r["class_ids"], r["scores"], r['masks'])
    APs.append(AP)

print("mAP: ", np.mean(APs))
