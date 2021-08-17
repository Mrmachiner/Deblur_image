# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'GUI/demo.ui'
#
# Created by: PyQt5 UI code generator 5.9.2
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import Qt, pyqtSignal, QTimer
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
import sys
import tkinter
sys.path.append("/home/minhhoang/Desktop/Deblur_image/scripts/")
# import model
import click
import os
import matplotlib.pyplot as plt
import math
from deblurgan.model import generator_model, generator_model_paper
from deblurgan.utils import load_image, deprocess_image, preprocess_image, preprocess_image_no_resize
from compare_model import compareModel, Yolo
import cv2
def is_an_image_file(filename):
    IMAGE_EXTENSIONS = ['.png', '.jpg', '.jpeg']
    for ext in IMAGE_EXTENSIONS:
        if ext in filename:
            return True
    return False

# ===================== CLASE QLabelClickable ======================

class QLabelClickable(QLabel):
    clicked = pyqtSignal(str)
    
    def __init__(self, parent=None):
        super(QLabelClickable, self).__init__(parent)

    def mousePressEvent(self, event):
        self.ultimo = "Clic"
    
    def mouseReleaseEvent(self, event):
        if self.ultimo == "Clic":
            QTimer.singleShot(QApplication.instance().doubleClickInterval(),
                              self.performSingleClickAction)
        else:
            # Realizar acción de doble clic.
            self.clicked.emit(self.ultimo)
    
    def mouseDoubleClickEvent(self, event):
        self.ultimo = "Doble Clic"
    
    def performSingleClickAction(self):
        if self.ultimo == "Clic":
            self.clicked.emit(self.ultimo)

class Ui_MainWindow(object):
    def __init__(self):
        super().__init__()
        self.model_me = compareModel()
        self.mode_original = compareModel(key="paper")
        self.yolo = Yolo("yolov5/yolov5s.pt", 0.25)
        self.display_width = 491
        self.display_height = 271
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1513, 812)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.pushButton_open = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_open.setGeometry(QtCore.QRect(510, 330, 89, 31))
        self.pushButton_open.setObjectName("pushButton_open")
        self.label_source = QtWidgets.QLabel(self.centralwidget)
        self.label_source.setGeometry(QtCore.QRect(210, 310, 101, 17))
        self.label_source.setObjectName("label_source")
        self.label_gen = QtWidgets.QLabel(self.centralwidget)
        self.label_gen.setGeometry(QtCore.QRect(710, 310, 121, 17))
        self.label_gen.setObjectName("label_gen")
        self.label_paper = QtWidgets.QLabel(self.centralwidget)
        self.label_paper.setGeometry(QtCore.QRect(1220, 310, 121, 17))
        self.label_paper.setObjectName("label_paper")

        self.img_source = QLabelClickable(self.centralwidget)
        self.img_source.setGeometry(QtCore.QRect(10, 10, 491, 271))
        self.img_source.setFrameShape(QtWidgets.QFrame.Box)
        self.img_source.setText("")
        self.img_source.setAlignment(QtCore.Qt.AlignCenter)
        self.img_source.setObjectName("img_source")

        self.img_gen = QLabelClickable(self.centralwidget)
        self.img_gen.setGeometry(QtCore.QRect(510, 10, 491, 271))
        self.img_gen.setFrameShape(QtWidgets.QFrame.Box)
        self.img_gen.setText("")
        self.img_gen.setObjectName("img_gen")

        self.img_paper = QLabelClickable(self.centralwidget)
        self.img_paper.setGeometry(QtCore.QRect(1010, 10, 491, 271))
        self.img_paper.setFrameShape(QtWidgets.QFrame.Box)
        self.img_paper.setText("")
        self.img_paper.setObjectName("img_paper")

        self.img_paper_yolo = QLabelClickable(self.centralwidget)
        self.img_paper_yolo.setGeometry(QtCore.QRect(1010, 370, 491, 271))
        self.img_paper_yolo.setFrameShape(QtWidgets.QFrame.Box)
        self.img_paper_yolo.setText("")
        self.img_paper_yolo.setObjectName("img_paper_yolo")

        self.img_gen_yolo = QLabelClickable(self.centralwidget)
        self.img_gen_yolo.setGeometry(QtCore.QRect(510, 370, 491, 271))
        self.img_gen_yolo.setFrameShape(QtWidgets.QFrame.Box)
        self.img_gen_yolo.setText("")
        self.img_gen_yolo.setObjectName("img_gen_yolo")

        self.img_source_yolo = QLabelClickable(self.centralwidget)
        self.img_source_yolo.setGeometry(QtCore.QRect(10, 370, 491, 271))
        self.img_source_yolo.setFrameShape(QtWidgets.QFrame.Box)
        self.img_source_yolo.setText("")
        self.img_source_yolo.setObjectName("img_source_yolo")

        self.pushButton_yolo = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_yolo.setGeometry(QtCore.QRect(910, 330, 89, 31))
        self.pushButton_yolo.setObjectName("pushButton_yolo")
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(10, 660, 491, 81))
        self.label.setFrameShape(QtWidgets.QFrame.Box)
        self.label.setText("")
        self.label.setObjectName("label")
        self.label_2 = QtWidgets.QLabel(self.centralwidget)
        self.label_2.setGeometry(QtCore.QRect(510, 660, 491, 81))
        self.label_2.setFrameShape(QtWidgets.QFrame.Box)
        self.label_2.setText("")
        self.label_2.setObjectName("label_2")
        self.label_3 = QtWidgets.QLabel(self.centralwidget)
        self.label_3.setGeometry(QtCore.QRect(1010, 660, 491, 81))
        self.label_3.setFrameShape(QtWidgets.QFrame.Box)
        self.label_3.setText("")
        self.label_3.setObjectName("label_3")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1513, 22))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
        self.img = ""
        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("DeblurGAN vs DeblurGANv2", "DeblurGAN vs DeblurGANv2"))
        self.pushButton_open.setText(_translate("MainWindow", "Open Image"))
        self.label_source.setText(_translate("MainWindow", "Image Source"))
        self.label_gen.setText(_translate("MainWindow", "Image Generator"))
        self.label_paper.setText(_translate("MainWindow", "Image Paper"))
        self.pushButton_yolo.setText(_translate("MainWindow", "Yolo"))
        self.pushButton_open.clicked.connect(self.pushButton_handler)
        self.pushButton_yolo.clicked.connect(self.button_yolo)
    def pushButton_handler(self):
        self.open_diablog_box()
    def button_yolo(self):
        self.process(key="Yolo")
    
    def show_img_source(self):
        image_s = cv2.imread("image_label/img_s.png")
        image_s = cv2.cvtColor(image_s, cv2.COLOR_BGR2RGB)
        cv2.imshow("Img_Source", image_s)

    def show_img_gen(self):
        image_g = cv2.imread("image_label/img_m.png")
        image_g = cv2.cvtColor(image_g, cv2.COLOR_BGR2RGB)
        cv2.imshow("Img_Gen", image_g)
        #cv2.waitKey()
        # plt.imshow(self.out_me)
        # plt.show()
    
    def show_img_paper(self):
        image_p = cv2.imread("image_label/img_p.png")
        image_p = cv2.cvtColor(image_p, cv2.COLOR_BGR2RGB)
        cv2.imshow("Img_Paper", image_p)
        #cv2.waitKey()
        # plt.imshow(self.out_paper)
        # plt.show()
    
    def show_img_source_yolo(self):
        image_s_y = cv2.cvtColor(self.source_yolo, cv2.COLOR_BGR2RGB)
        cv2.imshow("Img_Source_Yolo", image_s_y)
        #cv2.waitKey()
        # plt.imshow(self.source_yolo)
        # plt.show()
    
    def show_img_gen_yolo(self):
        image_m_y = cv2.cvtColor(self.me_yolo, cv2.COLOR_BGR2RGB)
        cv2.imshow("Img_Gen_Yolo", image_m_y)
        #cv2.waitKey()
        # plt.imshow(self.img_gen_yolo)
        # plt.show()
    
    def show_img_paper_yolo(self):
        image_p_y = cv2.cvtColor(self.paper_yolo, cv2.COLOR_BGR2RGB)
        cv2.imshow("Img_Paper_Yolo", image_p_y)
        #cv2.waitKey()
        # plt.imshow(self.paper_yolo)
        # plt.show()
    
    def conver_img(self, image):
        h, w, ch = image.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QtGui.QImage(image.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
        p = convert_to_Qt_format.scaled(self.display_width, self.display_height, Qt.KeepAspectRatio)
        return p
    
    def open_diablog_box(self):
        super(Ui_MainWindow, self)
        check = False
        filename = QtWidgets.QFileDialog.getOpenFileName()
        print(filename)
        if is_an_image_file(filename[0]):
            self.img = cv2.imread(filename[0])
            self.img = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)

            p = self.conver_img(self.img)
            # self.label.setPixmap(QPixmap.fromImage(p))
            self.img_source.setPixmap(QPixmap.fromImage(p))
            
            # self.gridLayout = QtWidgets.QGridLayout(self) 
            # self.gridLayout.addWidget(self.imgLabel, 0, 0, 1, 1)
            # plt.imshow(img)
            # plt.show()
            check = True
        else:
            msg = QtWidgets.QMessageBox()
            msg.setIcon(QtWidgets.QMessageBox.Information)
            msg.setText("Please open file .jpg .png .jpeg !!!")
            msg.setWindowTitle("Erro!!!")
            msg.exec_()
        if check:
            self.process()
    
    def process(self, key="GAN"):
        if key == "GAN":
            self.out_me = self.model_me.implement(self.img)
            self.out_paper = self.mode_original.paper(self.img)
            
            cv2.imwrite("image_label/img_s.png", self.img)
            cv2.imwrite("image_label/img_p.png", self.out_paper)
            cv2.imwrite("image_label/img_m.png", self.out_me)
            # cv2.imwrite("img_s_l.png",self.img_s_l)

            print("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaa", self.out_paper.shape)
            
            self.push_img("GAN")
        elif key == "Yolo":
            if self.img == "":
                msg = QtWidgets.QMessageBox()
                msg.setIcon(QtWidgets.QMessageBox.Information)
                msg.setText("Please open file .jpg .png .jpeg !!!")
                msg.setWindowTitle("Erro!!!")
                msg.exec_()
            else:
                self.source_yolo, self.str_source = self.yolo.detect(self.img)
                
                self.me_yolo, self.str_me = self.yolo.detect(self.out_me)
                
                self.paper_yolo, self.str_parer = self.yolo.detect(self.out_paper)
                self.push_img("Yolo")
    
    def push_img(self, key="GAN"):
        if key == "GAN":
            c_out_me = self.conver_img(self.out_me)
            c_out_paper = self.conver_img(self.out_paper)

            self.img_gen.setPixmap(QPixmap.fromImage(c_out_me))
            self.img_paper.setPixmap(QPixmap.fromImage(c_out_paper))

            # self.img_gen.clicked.connect(self.show_img_gen)
            # self.img_paper.clicked.connect(self.show_img_paper)
            # self.img_source.clicked.connect(self.show_img_source)
        elif key == "Yolo":
            c_source_yolo = self.conver_img(self.source_yolo)
            c_me_yolo = self.conver_img(self.me_yolo)
            c_paper_yolo = self.conver_img(self.paper_yolo)

            self.img_source_yolo.setPixmap(QPixmap.fromImage(c_source_yolo))
            self.img_gen_yolo.setPixmap(QPixmap.fromImage(c_me_yolo))
            self.img_paper_yolo.setPixmap(QPixmap.fromImage(c_paper_yolo))

            self.label.setText(self.str_source)
            self.label_2.setText(self.str_me)
            self.label_3.setText(self.str_parer)
            self.img_source_yolo.clicked.connect(self.show_img_source_yolo)
            self.img_gen_yolo.clicked.connect(self.show_img_gen_yolo)
            self.img_paper_yolo.clicked.connect(self.show_img_paper_yolo)
            self.img_gen.clicked.connect(self.show_img_gen)
            self.img_paper.clicked.connect(self.show_img_paper)
            self.img_source.clicked.connect(self.show_img_source)


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())

