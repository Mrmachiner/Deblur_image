# # -*- coding: utf-8 -*-

# # Form implementation generated from reading ui file 'demo.ui'
# #
# # Created by: PyQt5 UI code generator 5.9.2
# #
# # WARNING! All changes made in this file will be lost!

# from PyQt5 import QtCore, QtGui, QtWidgets
# from PyQt5.QtCore import Qt, pyqtSignal
# from PyQt5.QtGui import *
# from PyQt5.QtWidgets import *
# import matplotlib.pyplot as plt
# import sys
# import cv2
# def is_an_image_file(filename):
#     IMAGE_EXTENSIONS = ['.png', '.jpg', '.jpeg']
#     for ext in IMAGE_EXTENSIONS:
#         if ext in filename:
#             return True
#     return False


        
# class myImgLabel(QtWidgets.QLabel):
#     def __init__(self, parent=None):
#         super(myImgLabel, self).__init__(parent)
#         f = QFont("ZYSong18030",10) # Set the font, font size
#         self.setFont(f) # After the event is not defined, the two sentences are deleted or commented out.
    
#     '''Reload the mouse click event (click) '''
#     def mousePressEvent(self, event):
#         if event.buttons () == QtCore.Qt.LeftButton: # left button pressed
#             self.setText ("Click the left mouse button for the event: define it yourself")
#             print("Click the left mouse button") # response test statement
#         elif event.buttons () == QtCore.Qt.RightButton: # right click
#             self.setText ("Click the right mouse button for the event: define it yourself")
#             print("right click") # response test statement
#         elif event.buttons () == QtCore.Qt.MidButton: #  Press
#             self.setText ("Click the middle mouse button for the event: define it yourself")
#             print("click the middle mouse button") # response test statement
#         elif event.buttons () == QtCore.Qt.LeftButton | QtCore.Qt.RightButton: # Left and right buttons simultaneously pressed
#             self.setText ("Also click the left and right mouse button event: define it yourself")
#             print("Click the left and right mouse button") # response test statement
#         elif event.buttons () == QtCore.Qt.LeftButton | QtCore.Qt.MidButton: # Left middle button simultaneously pressed
#             self.setText ("Also click the middle left mouse button event: define it yourself")
#             print("Click the left middle button") # response test statement
#         elif event.buttons () == QtCore.Qt.MidButton | QtCore.Qt.RightButton: #  
#             self.setText ("Also click the middle right mouse button event: define it yourself")
#             print("click the middle right button") # response test statement
#         elif event.buttons () == QtCore.Qt.LeftButton | QtCore.Qt.MidButton \
#                            | QtCore.Qt.RightButton: # Left and right button simultaneously pressed
#             self.setText ("Also click the left mouse button right event: define it yourself")
#             print("Click the left and right mouse button") # response test statement
 
    
#     '''Overload the wheel scrolling event '''
#     def wheelEvent(self, event):
#  # if event.delta() > 0: # Roller up, PyQt4
#         # This function has been deprecated, use pixelDelta() or angleDelta() instead.
#         Angle=event.angleDelta() / 8 # Returns the QPoint object, the value of the wheel, in 1/8 degrees
#         angleX=angle.x() # The distance rolled horizontally (not used here)
#         angleY=angle.y() # The distance that is rolled vertically
#         if angleY > 0:
#             self.setText("Scroll up event: define itself")
#             print("mouse wheel scrolling") # response test statement
#         else: #roll down
#             self.setText("Scroll down event: define itself")
#             print("mouse wheel down") # response test statement
        
#     '''Overload the mouse double click event '''
#     def mouseDoubieCiickEvent(self, event):
#     # if event.buttons () == QtCore.Qt.LeftButton: # Left button pressed
#     # self.setText ("Double-click the left mouse button function: define it yourself")
#         self.setText ("mouse double click event: define itself")
 
#     '''Reload the mouse button release event '''
#     def mouseReleaseEvent(self, event):
#                  self.setText("mouse release event: define itself")
#                  print("mouse release") # response test statement
 


# class MyWindow(QtWidgets.QWidget):
#     def __init__(self):
#         super(MyWindow, self).__init__()
#         self.imgLabel = myImgLabel() # declare imgLabel
#         self.image = QImage() # declare new img
#         # if self.image.load("/home/minhhoang/Desktop/Deblur_image/input/000001.png"): # If the image is loaded, then
#         #         self.imgLabel.setPixmap(QPixmap.fromImage(self.image)) # Display image
#         # self.gridLayout = QtWidgets.QGridLayout (self) # Layout settings
#         # self.gridLayout.addWidget (self.imgLabel, 0, 0, 1, 1) 
#     def setupUi(self, MainWindow):
#         MainWindow.setObjectName("MainWindow")
#         MainWindow.resize(800, 600)
#         self.centralwidget = QtWidgets.QWidget(MainWindow)
#         self.centralwidget.setObjectName("centralwidget")
#         self.pushButton = QtWidgets.QPushButton(self.centralwidget)
#         self.pushButton.setGeometry(QtCore.QRect(270, 320, 89, 25))
#         self.pushButton.setObjectName("pushButton")
        
#         self.disply_width = 320
#         self.display_height = 180
#         self.label = QtWidgets.QLabel(self.centralwidget)
#         self.label.setGeometry(QtCore.QRect(10, 10, self.disply_width, self.display_height))
#         self.label.setObjectName("label")

#         self.retranslateUi(MainWindow)
#         QtCore.QMetaObject.connectSlotsByName(MainWindow)
#     def retranslateUi(self, MainWindow):
#         _translate = QtCore.QCoreApplication.translate
#         MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
#         self.pushButton.setText(_translate("MainWindow", "Open Folder"))
#         self.label.setText(_translate("MainWindow", "pic1"))
#         self.pushButton.clicked.connect(self.pushButton_handler)

#     def pushButton_handler(self):
#         self.open_diablog_box()

#     def open_diablog_box(self):
#         check = False
#         filename = QtWidgets.QFileDialog.getOpenFileName()
#         print(filename)
#         if is_an_image_file(filename[0]):
#             img = cv2.imread(filename[0])
            
#             h, w, ch = img.shape
#             bytes_per_line = ch * w
#             convert_to_Qt_format = QtGui.QImage(img.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
#             p = convert_to_Qt_format.scaled(self.disply_width, self.display_height, Qt.KeepAspectRatio)
#             self.label.setPixmap(QPixmap.fromImage(p))
#             check = True
#         else:
#             msg = QtWidgets.QMessageBox()
#             msg.setIcon(QtWidgets.QMessageBox.Information)
#             msg.setText("Please open file .jpg .png .jpeg !!!")
#             msg.setWindowTitle("Erro!!!")
#             msg.exec_()
#         if check:
#             # start = time.time()

#             # self.process(filename[0])
#             # process_time = time.time() - start
#             # print("abcd", process_time)
#             print("abcd")
#     # def process(self, path):
#     #     pre, time, lst_img = self.pre.predict_img(path)
#     #     print(pre)
# class Ui_MainWindow(object):
#     def setupUi(self, MainWindow):
#         MainWindow.setObjectName("MainWindow")
#         MainWindow.resize(800, 600)
#         self.centralwidget = QtWidgets.QWidget(MainWindow)
#         self.centralwidget.setObjectName("centralwidget")
#         self.pushButton = QtWidgets.QPushButton(self.centralwidget)
#         self.pushButton.setGeometry(QtCore.QRect(270, 320, 89, 25))
#         self.pushButton.setObjectName("pushButton")
        
#         self.display_width = 491
#         self.display_height = 271
#         self.label = QtWidgets.QLabel(self.centralwidget)
#         self.label.setGeometry(QtCore.QRect(10, 10, self.disply_width, self.display_height))
#         self.label.setObjectName("label")
#         MainWindow.setCentralWidget(self.centralwidget)
#         self.menubar = QtWidgets.QMenuBar(MainWindow)
#         self.menubar.setGeometry(QtCore.QRect(0, 0, 800, 22))
#         self.menubar.setObjectName("menubar")
#         MainWindow.setMenuBar(self.menubar)
#         self.statusbar = QtWidgets.QStatusBar(MainWindow)
#         self.statusbar.setObjectName("statusbar")
#         MainWindow.setStatusBar(self.statusbar)

#         self.retranslateUi(MainWindow)
#         QtCore.QMetaObject.connectSlotsByName(MainWindow)

#     def retranslateUi(self, MainWindow):
#         _translate = QtCore.QCoreApplication.translate
#         MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
#         self.pushButton.setText(_translate("MainWindow", "Open Folder"))
#         self.label.setText(_translate("MainWindow", "pic1"))
#         self.pushButton.clicked.connect(self.pushButton_handler)

#     def pushButton_handler(self):
#         self.open_diablog_box()

#     def open_diablog_box(self):
#         super(Ui_MainWindow, self)
#         check = False
#         filename = QtWidgets.QFileDialog.getOpenFileName()
#         print(filename)
#         if is_an_image_file(filename[0]):
#             img = cv2.imread(filename[0])
            
#             h, w, ch = img.shape
#             bytes_per_line = ch * w
#             convert_to_Qt_format = QtGui.QImage(img.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
#             p = convert_to_Qt_format.scaled(self.disply_width, self.display_height, Qt.KeepAspectRatio)
#             # self.label.setPixmap(QPixmap.fromImage(p))
#             self.imgLabel = myImgLabel()
#             self.imgLabel.setPixmap(QPixmap.fromImage(p))
#             # self.gridLayout = QtWidgets.QGridLayout(self) 
#             # self.gridLayout.addWidget(self.imgLabel, 0, 0, 1, 1)
#             # imgplot = plt.imshow(img)
#             # plt.show()
#             check = True
#         else:
#             msg = QtWidgets.QMessageBox()
#             msg.setIcon(QtWidgets.QMessageBox.Information)
#             msg.setText("Please open file .jpg .png .jpeg !!!")
#             msg.setWindowTitle("Erro!!!")
#             msg.exec_()
#         if check:
#             # start = time.time()

#             # self.process(filename[0])
#             # process_time = time.time() - start
#             # print("abcd", process_time)
#             print("abcd")
#     # def process(self, path):
#     #     pre, time, lst_img = self.pre.predict_img(path)
#     #     print(pre)
#     #     print(time)
#         # print(filename[0])
# if __name__ == "__main__":
#     app = QtWidgets.QApplication(sys.argv)
#     myshow = MyWindow()
#     myshow.setupUi(myshow)
#     myshow.show()
#     sys.exit (app.exec_())

#     # app = QtWidgets.QApplication(sys.argv)
#     # MainWindow = QtWidgets.QMainWindow()
#     # ui = Ui_MainWindow()
#     # ui.setupUi(MainWindow)
#     # MainWindow.show()
#     # sys.exit(app.exec_())

