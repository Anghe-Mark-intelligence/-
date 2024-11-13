import tensorflow as tf
import os
import glob
import numpy as np
import cv2
from utils.unet import Unet
import tkinter as tk
from tkinter import filedialog
from tkinter import Label, Button
from PIL import Image, ImageTk
import shutil
import albumentations as A

# Directory paths
data_dir = r'C:\Users\Administrator\Desktop\heangcomputervision\第七次实验\data\DATEPREPROCESS'
model_dir = r'C:\Users\Administrator\Desktop\heangcomputervision\第七次实验\model'
train_dir = os.path.join(data_dir, 'train')
test_dir = os.path.join(data_dir, 'test')

# Data augmentation using Albumentations
transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.2),
    A.Rotate(limit=30, p=0.5),
    A.Resize(256, 256)
])

# Split data into train and test sets if not already done
if not os.path.exists(train_dir) or not os.path.exists(test_dir):
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    images = glob.glob(os.path.join(data_dir, '*.png'))
    images += glob.glob(os.path.join(data_dir, '*.jpg'))
    np.random.shuffle(images)

    split_index = int(0.8 * len(images))
    train_images = images[:split_index]
    test_images = images[split_index:]

    for img in train_images:
        shutil.copy(img, os.path.join(train_dir, os.path.basename(img)))
    for img in test_images:
        shutil.copy(img, os.path.join(test_dir, os.path.basename(img)))

# Load and preprocess the data
def load_image(image_path):
    image = cv2.imread(image_path)
    augmented = transform(image=image)
    image = augmented["image"] / 255.0
    return image

def load_mask(mask_path):
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    mask = cv2.resize(mask, (256, 256))
    mask = mask / 255.0
    mask = np.expand_dims(mask, axis=-1)
    return mask

train_images = []
train_masks = []
test_images = []
test_masks = []

for img_path in glob.glob(os.path.join(train_dir, '*.jpg')):
    base_name = os.path.basename(img_path).split('.')[0]
    mask_path_1 = os.path.join(train_dir, f'{base_name}_1stHO.png')
    mask_path_2 = os.path.join(train_dir, f'{base_name}_2ndHO.png')

    if os.path.exists(mask_path_1):
        train_images.append(load_image(img_path))
        train_masks.append(load_mask(mask_path_1))
    elif os.path.exists(mask_path_2):
        train_images.append(load_image(img_path))
        train_masks.append(load_mask(mask_path_2))

for img_path in glob.glob(os.path.join(test_dir, '*.jpg')):
    base_name = os.path.basename(img_path).split('.')[0]
    mask_path_1 = os.path.join(test_dir, f'{base_name}_1stHO.png')
    mask_path_2 = os.path.join(test_dir, f'{base_name}_2ndHO.png')

    if os.path.exists(mask_path_1):
        test_images.append(load_image(img_path))
        test_masks.append(load_mask(mask_path_1))
    elif os.path.exists(mask_path_2):
        test_images.append(load_image(img_path))
        test_masks.append(load_mask(mask_path_2))

# Convert to numpy arrays
train_images = np.array(train_images)
train_masks = np.array(train_masks)
test_images = np.array(test_images)
test_masks = np.array(test_masks)

# Define Dice Loss
def dice_loss(y_true, y_pred):
    y_true_f = tf.keras.backend.flatten(y_true)
    y_pred_f = tf.keras.backend.flatten(y_pred)
    intersection = tf.keras.backend.sum(y_true_f * y_pred_f)
    return 1 - (2. * intersection + 1) / (tf.keras.backend.sum(y_true_f) + tf.keras.backend.sum(y_pred_f) + 1)

# Define the dataset
batch_size = 4
dataset = tf.data.Dataset.from_tensor_slices((train_images, train_masks)).batch(batch_size).repeat()

# Define model parameters
data_format = 'channels_last'
classes = 1  # Assuming binary segmentation
transpose_conv = True

# Initialize the U-Net model
model = Unet(data_format=data_format, classes=classes, transpose_conv=transpose_conv)

# Compile the model using Dice Loss
model.compile(optimizer='adam', loss=dice_loss, metrics=['accuracy'])

# Train the model
os.makedirs(model_dir, exist_ok=True)
model.fit(
    dataset,
    epochs=1400,  # Increased epochs for better training
    steps_per_epoch=len(train_images) // batch_size,
    callbacks=[
        tf.keras.callbacks.TensorBoard(log_dir=model_dir)
    ]
)

# Save the entire model in SavedModel format
model_save_path = os.path.join(model_dir, 'saved_model')
model.save(model_save_path, save_format='tf')

# Evaluate the model
eval_dataset = tf.data.Dataset.from_tensor_slices((test_images, test_masks)).batch(batch_size)
loss, accuracy = model.evaluate(eval_dataset)
print(f'Test Loss: {loss}, Test Accuracy: {accuracy}')
