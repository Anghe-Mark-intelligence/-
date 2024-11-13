# svm.py 文件

from PIL import Image
import os
import sys
import numpy as np
from sklearn import svm
import joblib
import warnings
warnings.filterwarnings('ignore')

# 步骤二: 获取所有指定路径下的.jpg 文件
def get_file_list(path):
    return [os.path.join(path, f) for f in os.listdir(path) if f.endswith(".jpg")]

# 步骤三: 获取图像名称
def get_img_name_str(imgPath):
    return imgPath.split(os.path.sep)[-1]

# 步骤四: 将16像素*16像素的图像数据转换成1*256的NumPy向量
def img2vector(imgFile):
    img = Image.open(imgFile).convert('L')  # 转为灰度图
    img_arr = np.array(img, 'i')            # 转为numpy数组
    img_normalization = np.round(img_arr / 255.0)  # 归一化处理
    img_arr2 = np.reshape(img_normalization, (1, -1))  # 转换为 1x256 向量
    return img_arr2

# 步骤五: 将图像文件转化为向量并获取标签
def read_and_convert(imgFilelist):
    dataLabel = []  # 存放类标签
    dataNum = len(imgFilelist)
    dataMat = np.zeros((dataNum, 256))  # 创建 dataNum * 256 的矩阵
    for i in range(dataNum):
        imgNameStr = imgFilelist[i]
        imgName = get_img_name_str(imgNameStr)  # 获取文件名
        classTag = imgName.split("_")[0]  # 获取类标签（数字）
        dataLabel.append(int(classTag))  # 转换为整数标签
        dataMat[i, :] = img2vector(imgNameStr)
    return dataMat, dataLabel

# 步骤六: 读取训练数据
def read_all_data(train_data_path):
    # 获取所有图像文件
    flist = get_file_list(train_data_path)
    # 转化为图像矩阵和标签
    dataMat, dataLabel = read_and_convert(flist)
    return dataMat, dataLabel

# 步骤七: 创建并训练 SVM 模型
def create_svm(dataMat, dataLabel, path, decision='ovr'):
    clf = svm.SVC(decision_function_shape=decision)
    rf = clf.fit(dataMat, dataLabel)
    joblib.dump(rf, path)  # 存储模型
    return clf

# 步骤八: 主函数处理
if __name__ == '__main__':
    print('正在运行模型，请稍等...')
    
    # 设置训练数据路径
    train_data_path = r'C:\Users\Administrator\Desktop\heangcomputervision\第四次实验\img_train_heang'
    
    # 模型存储路径
    model_path = r'C:\Users\Administrator\Desktop\heangcomputervision\第四次实验\svm.model'
    
    # 调用函数，获取图像矩阵和标签
    dataMat, dataLabel = read_all_data(train_data_path)

    # 训练并保存模型
    create_svm(dataMat, dataLabel, model_path, decision='ovr')
    
    print('模型训练和存储完成')
