import os
import cv2
import numpy as np
import argparse
import shutil
import glob

def parse_args():
    parser = argparse.ArgumentParser(description='Preprocess image data and split into train/test sets')
    parser.add_argument('-data_dir', type=str, required=False, default=r'C:\Users\Administrator\Desktop\heangcomputervision\第七次实验\data', help='Path to the data directory')
    parser.add_argument('-train_ratio', type=float, required=False, default=0.8, help='Ratio of training data')
    return parser.parse_args()

def gamma_correction(image, gamma):
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype('uint8')
    return cv2.LUT(image, table)

def preprocess_image(image):
    # Convert the color image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Normalize the image data
    normalized_image = cv2.normalize(gray_image, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    
    # Histogram equalization
    equalized_image = cv2.equalizeHist(normalized_image)
    
    # Gamma correction, assuming gamma value is 1.2
    gamma = 1.2
    gamma_corrected = gamma_correction(equalized_image, gamma)
    
    return gamma_corrected

def main():
    args = parse_args()
    data_dir = args.data_dir
    train_ratio = args.train_ratio

    # Check if the data directory exists
    if not os.path.exists(data_dir):
        print(f"Data directory {data_dir} does not exist.")
        return

    # Create DATEPREPROCESS directory
    date_preprocess_dir = os.path.join(data_dir, 'DATEPREPROCESS')
    os.makedirs(date_preprocess_dir, exist_ok=True)

    # Initialize a counter for processed images
    processed_count = 0

    # Traverse the data directory and preprocess images
    for root, dirs, files in os.walk(data_dir):
        # Skip the DATEPREPROCESS directory to avoid infinite loops
        if 'DATEPREPROCESS' in root:
            continue

        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_path = os.path.join(root, file)

                # Skip mask images (assuming mask filenames contain "HO")
                if "HO" in file:
                    continue

                # Read the original image
                image = cv2.imread(image_path)

                if image is None:
                    print(f"Warning: Failed to read {image_path}. Skipping this file.")
                    continue

                # Process the image
                processed_image = preprocess_image(image)

                # Save the processed image to DATEPREPROCESS directory
                save_path_datepreprocess = os.path.join(date_preprocess_dir, file)
                cv2.imwrite(save_path_datepreprocess, processed_image)

                # Increment the processed count
                processed_count += 1

    print(f"Processing complete. Total images processed: {processed_count}")

    # Split processed data into train and test sets
    processed_images = glob.glob(os.path.join(date_preprocess_dir, '*.png'))
    processed_images += glob.glob(os.path.join(date_preprocess_dir, '*.jpg'))
    np.random.shuffle(processed_images)

    split_index = int(train_ratio * len(processed_images))
    train_images = processed_images[:split_index]
    test_images = processed_images[split_index:]

    # Create train and test directories
    train_dir = os.path.join(data_dir, 'train')
    test_dir = os.path.join(data_dir, 'test')
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    # Copy images to train and test directories
    for img in train_images:
        shutil.copy(img, os.path.join(train_dir, os.path.basename(img)))
    for img in test_images:
        shutil.copy(img, os.path.join(test_dir, os.path.basename(img)))

    # Process and copy corresponding masks
    for root, dirs, files in os.walk(data_dir):
        if 'DATEPREPROCESS' in root:
            continue
        for file in files:
            if "HO" in file and file.lower().endswith(('.png', '.jpg', '.jpeg')):
                mask_path = os.path.join(root, file)
                base_name = file.split('_')[0]

                # Find the corresponding processed image in train/test
                if os.path.exists(os.path.join(train_dir, f"{base_name}.jpg")):
                    shutil.copy(mask_path, os.path.join(train_dir, f"{base_name}_mask.png"))
                elif os.path.exists(os.path.join(test_dir, f"{base_name}.jpg")):
                    shutil.copy(mask_path, os.path.join(test_dir, f"{base_name}_mask.png"))

    print(f"Data split complete. Training images: {len(train_images)}, Testing images: {len(test_images)}")

if __name__ == '__main__':
    main()
