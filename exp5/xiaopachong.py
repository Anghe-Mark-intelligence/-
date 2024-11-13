import os

# 指定路径
path = r'C:\Users\Administrator\Desktop\heangcomputervision\第五次实验\PlantVillage\Plant_leave_diseases_dataset_with_augmentation'

# 列出文件夹
folders = [folder for folder in os.listdir(path) if os.path.isdir(os.path.join(path, folder))]

# 输出文件夹名称
for folder in folders:
    print(folder)

# 确保正确数量
if len(folders) == 39:
    print(f"总共找到 {len(folders)} 个文件夹。")
else:
    print(f"警告：找到的文件夹数量为 {len(folders)}，而不是 39 个。")
