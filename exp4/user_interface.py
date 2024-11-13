import sys
from PyQt5.QtWidgets import QFileDialog, QMainWindow, QApplication
from PyQt5 import QtCore, QtGui, QtWidgets
import os
import joblib
# 调用自己创建的类
import svm

# 步骤二: 创建类，完成可视化窗口的初始化
class Ui_Dialog(object):
    def setupUi(self, Dialog):
        Dialog.setObjectName("Dialog")
        # 设置窗口大小
        Dialog.resize(645, 475)
        # 设置“打开图像”按钮
        self.pushButton = QtWidgets.QPushButton(Dialog)
        self.pushButton.setGeometry(QtCore.QRect(230, 340, 141, 41))
        self.pushButton.setAutoDefault(False)
        self.pushButton.setObjectName("pushButton")
        # 设置“显示图像”标签
        self.label = QtWidgets.QLabel(Dialog)
        self.label.setGeometry(QtCore.QRect(220, 50, 191, 221))
        self.label.setWordWrap(False)
        self.label.setObjectName("label")
        # 设置文本编辑区
        self.textEdit = QtWidgets.QTextEdit(Dialog)
        self.textEdit.setGeometry(QtCore.QRect(220, 280, 191, 41))
        self.textEdit.setObjectName("textEdit")

        self.retranslateUi(Dialog)
        QtCore.QMetaObject.connectSlotsByName(Dialog)

    # 创建窗口设置
    def retranslateUi(self, Dialog):
        _translate = QtCore.QCoreApplication.translate
        Dialog.setWindowTitle(_translate("Dialog", "何昂202210310219手写体识别"))
        self.pushButton.setText(_translate("Dialog", "打开图像"))
        self.label.setText(_translate("Dialog", "何昂202210310219显示图像"))

# 步骤三: 创建类，完成测试集图像验证功能
class MyWindow(QMainWindow, Ui_Dialog):
    # 初始化数据
    def __init__(self, parent=None):
        super(MyWindow, self).__init__(parent)
        self.setupUi(self)
        self.pushButton.clicked.connect(self.openImage)  # 点击事件，开启下面的函数

    # 点击事件函数
    def openImage(self):
        # 点击“打开图像”按钮时，打开文件对话框选择图像
        imgName, imgType = QFileDialog.getOpenFileName(self, "打开图像", "", "Image Files (*.jpg *.png)")
        
        # 显示选择的图像
        if imgName:
            png = QtGui.QPixmap(imgName).scaled(self.label.width(), self.label.height())
            self.label.setPixmap(png)
            self.textEdit.setText(imgName)
            
            # 加载 SVM 模型并预测图像类别
            path = sys.path[0]
            model_path = os.path.join(path, r'svm.model')
            clf = joblib.load(model_path)

            # 使用 svm.py 中的 img2vector 函数将图像转为向量
            dataMat = svm.img2vector(imgName)
            
            # 进行预测
            preResult = clf.predict(dataMat)

            # 在文本框中显示预测结果
            self.textEdit.setReadOnly(True)
            self.textEdit.setStyleSheet("color:red")
            self.textEdit.setAlignment(QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter)
            self.textEdit.setFontPointSize(9)
            self.textEdit.setText("预测的结果是: ")
            self.textEdit.append(str(preResult[0]))  # 显示预测标签

# 步骤四: 主函数处理
if __name__ == '__main__':
    app = QApplication(sys.argv)
    myWin = MyWindow()
    myWin.show()
    sys.exit(app.exec_())
