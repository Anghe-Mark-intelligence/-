import tensorflow as tf
import os
import cv2
import numpy as np
from utils.unet import Unet
import tkinter as tk
from tkinter import filedialog
from tkinter import Label, Button
from PIL import Image, ImageTk
import glob

def dice_loss(y_true, y_pred, smooth=1e-6):
    intersection = tf.reduce_sum(y_true * y_pred)
    return 1 - (2. * intersection + smooth) / (tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) + smooth)

# Directory paths
model_dir = r'C:\Users\Administrator\Desktop\heangcomputervision\第七次实验\model\saved_model'

# Load the saved model
if not os.path.exists(model_dir):
    print("Warning: No saved model found in the specified directory. Please ensure the model has been trained and saved.")
    exit()

try:
    model = tf.keras.models.load_model(model_dir, custom_objects={'dice_loss': dice_loss})
except Exception as e:
    print(f"Error loading model: {e}. This could be due to mismatched layers or incorrect model definition.")
    exit()

# GUI for selecting and displaying images
def open_and_evaluate_image():
    file_path = filedialog.askopenfilename()
    if not file_path:
        return

    # Load and preprocess the selected image
    image = cv2.imread(file_path)
    original_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    processed_image = cv2.resize(original_image, (256, 256))
    processed_image = processed_image / 255.0
    processed_image = np.expand_dims(processed_image, axis=0)

    # Perform prediction using the model
    prediction = model.predict(processed_image)
    print(f"Prediction min: {prediction.min()}, max: {prediction.max()}")  # Print min and max values of prediction
    prediction = (prediction > 0.1).astype(np.uint8) * 255  # Lower threshold for better visualization

    # Resize prediction back to smaller size for better GUI visualization
    original_image_resized = cv2.resize(original_image, (200, 200))
    prediction_resized = cv2.resize(prediction[0, :, :, 0], (200, 200))

    # Apply color map for better visualization
    prediction_colored = cv2.applyColorMap(prediction_resized, cv2.COLORMAP_JET)

    # Display the original and segmented images
    original_image_pil = Image.fromarray(original_image_resized)
    prediction_pil = Image.fromarray(prediction_colored)

    original_image_tk = ImageTk.PhotoImage(original_image_pil)
    prediction_image_tk = ImageTk.PhotoImage(prediction_pil)

    label_original.configure(image=original_image_tk)
    label_original.image = original_image_tk
    label_prediction.configure(image=prediction_image_tk)
    label_prediction.image = prediction_image_tk

# Create the GUI window
root = tk.Tk()
root.title("U-Net Model Evaluation heang202210310219")
root.geometry("500x300")

# Create and place labels and buttons
label_original = Label(root, text="Original Image")
label_original.grid(row=0, column=0, padx=10, pady=10)
label_prediction = Label(root, text="Segmented Image")
label_prediction.grid(row=0, column=1, padx=10, pady=10)

button_open = Button(root, text="Open and Evaluate Image", command=open_and_evaluate_image)
button_open.grid(row=1, column=0, columnspan=2, pady=20)

# Run the GUI event loop
root.mainloop()
