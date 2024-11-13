import os

# Define the paths for the image and annotation directories
img_dir = 'C:\\Users\\Administrator\\Desktop\\heangcomputervision\\shiyan6\\laboro_tomato_DatasetNinja\\Train\\img'
ann_dir = 'C:\\Users\\Administrator\\Desktop\\heangcomputervision\\shiyan6\\laboro_tomato_DatasetNinja\\Train\\ann'

# Get file names in img directory with .jpg extension
img_files = sorted([f for f in os.listdir(img_dir) if f.endswith('.jpg')])

# Get file names in ann directory with .json extension
ann_files = sorted([f for f in os.listdir(ann_dir) if f.endswith('.json')])

# Print the lists of image and annotation files
print("Image files in img directory:")
for img_file in img_files:
    print(img_file)

print("\nAnnotation files in ann directory:")
for ann_file in ann_files:
    print(ann_file)

# Check matching files between img and ann directories
print("\nMatching result:")
for img_file in img_files:
    corresponding_ann_file = img_file.replace(".jpg", ".jpg.json")
    if corresponding_ann_file in ann_files:
        print(f"Match found for: {img_file} -> {corresponding_ann_file}")
    else:
        print(f"No matching annotation for: {img_file}")
