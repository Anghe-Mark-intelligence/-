import sys
import time
# 调用自己创建的类
import svm  
import os
import joblib
path = sys.path[0]# 获取模型位置
model_path = os.path.join(path, r'svm.model')
test_data_path = r'C:\Users\Administrator\Desktop\heangcomputervision\第四次实验\img_test_heang'# 加载测试集数据位置
tst = time.time()# 记录开始时间
clf = joblib.load(model_path)# 加载模型
tflist = svm.get_file_list(test_data_path)# 读取所有测试集图像
tdataMat, tdataLabel = svm.read_and_convert(tflist)# 将数据转化为图像矩阵和标签

print("测试集数据维度为: {0}, 标签数量: {1}".format(tdataMat.shape, len(tdataLabel)))# 打印测试集的维度信息
score_st = time.time()# 记录开始预测的时间
score = clf.score(tdataMat, tdataLabel)# 预测效果
score_et = time.time()# 记录结束预测的时间

print("何昂202210310219计算准确率花费 {:.6f} 秒.".format(score_et - score_st))# 打印预测结果
print("准确率: {:.6f}".format(score))
print("错误率: {:.6f}".format(1 - score))
tet = time.time()# 记录测试总耗时
print("测试总耗时 {:.6f} 秒.".format(tet - tst))
