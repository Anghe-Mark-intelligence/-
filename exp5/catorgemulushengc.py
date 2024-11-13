import json
import os

# 定义保存路径
save_path = r'C:\Users\Administrator\Desktop\heangcomputervision\第五次实验\PlantVillage\categories.json'

# 定义39个类别
categories = {
    "0": "Apple___Apple_scab",
    "1": "Apple___Black_rot",
    "2": "Apple___Cedar_apple_rust",
    "3": "Apple___healthy",
    "4": "Background_without_leaves",
    "5": "Blueberry___healthy",
    "6": "Cherry___healthy",
    "7": "Cherry___Powdery_mildew",
    "8": "Corn___Cercospora_leaf_spot Gray_leaf_spot",
    "9": "Corn___Common_rust",
    "10": "Corn___healthy",
    "11": "Corn___Northern_Leaf_Blight",
    "12": "Grape___Black_rot",
    "13": "Grape___Esca_(Black_Measles)",
    "14": "Grape___healthy",
    "15": "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)",
    "16": "Orange___Haunglongbing_(Citrus_greening)",
    "17": "Peach___Bacterial_spot",
    "18": "Peach___healthy",
    "19": "Pepper,_bell___Bacterial_spot",
    "20": "Pepper,_bell___healthy",
    "21": "Potato___Early_blight",
    "22": "Potato___healthy",
    "23": "Potato___Late_blight",
    "24": "Raspberry___healthy",
    "25": "Soybean___healthy",
    "26": "Squash___Powdery_mildew",
    "27": "Strawberry___healthy",
    "28": "Strawberry___Leaf_scorch",
    "29": "Tomato___Bacterial_spot",
    "30": "Tomato___Early_blight",
    "31": "Tomato___healthy",
    "32": "Tomato___Late_blight",
    "33": "Tomato___Leaf_Mold",
    "34": "Tomato___Septoria_leaf_spot",
    "35": "Tomato___Spider_mites Two-spotted_spider_mite",
    "36": "Tomato___Target_Spot",
    "37": "Tomato___Tomato_mosaic_virus",
    "38": "Tomato___Tomato_Yellow_Leaf_Curl_Virus"
}

# 将类别保存为 JSON 文件
with open(save_path, 'w') as json_file:
    json.dump(categories, json_file)

print(f"categories.json 文件已成功创建在 {save_path}！何昂202210310219")
