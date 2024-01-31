import sys

import cv2
import torch

from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QFileDialog, QVBoxLayout, QLabel, QWidget
from PyQt5.QtGui import QImageReader, QPixmap
from Application.Classifier import Classifier
from PIL import Image


class FileDialogExample(QMainWindow):
    def __init__(self):
        super().__init__()
        self.label = None
        self.layout = None
        self.file_name = ''
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.classifier = Classifier(device)
        self.initUI()

    def initUI(self):
        self.setGeometry(100, 100, 400, 200)
        self.setWindowTitle('File Dialog Example')

        self.layout = QVBoxLayout()
        self.label = QLabel(self)
        self.layout.addWidget(self.label)

        # 设置主窗口的中央部件为一个QWidget，用于容纳布局
        central_widget = QWidget(self)
        central_widget.setLayout(self.layout)
        self.setCentralWidget(central_widget)

        btn = QPushButton('Open Image', self)
        btn.clicked.connect(self.showDialog)

        btn_classification = QPushButton('Classify', self)
        btn_classification.clicked.connect(self.classify)

        self.layout.addWidget(btn)
        self.layout.addWidget(btn_classification)
    def showDialog(self):
        # 打开文件对话框，获取选择的文件路径
        self.file_name, _ = QFileDialog.getOpenFileName(self, 'Open file', '.',
                                                        'All Files (*);;Text Files (*.txt);;Image Files (*.png *.jpg *.gif)')

        image_reader = QImageReader(self.file_name)
        if image_reader.size().isValid():
            print(f'Selected file is an image: {self.file_name}')
            pixmap = QPixmap(self.file_name)

            self.label.setPixmap(pixmap)
            self.label.resize(pixmap.width(), pixmap.height())
        else:
            print(f'Selected file is not a valid image: {self.file_name}')

    def classify(self):
        image_reader = QImageReader(self.file_name)
        if image_reader.size().isValid():
            image = Image.open(self.file_name)
            res = self.classifier.classify(image)
            cv2.imwrite('res.jpg', res)
            pixmap = QPixmap('res.jpg')

            self.label.setPixmap(pixmap)
            self.label.resize(pixmap.width(), pixmap.height())


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = FileDialogExample()
    ex.show()
    sys.exit(app.exec_())
