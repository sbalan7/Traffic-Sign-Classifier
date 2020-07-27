from PyQt5.QtWidgets import (qApp, QLabel, QFrame, QAction, QWidget, QTextEdit, QGridLayout,
                             QPushButton, QMainWindow, QApplication, QFileDialog)
from PyQt5.QtGui import QIcon, QPixmap, QColor, QPalette
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import sys
import os


class tscGUI(QMainWindow):

    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        openAct = QAction(QIcon('open.png'), '&Open', self)
        openAct.setShortcut('Ctrl+O')
        openAct.setStatusTip('Open File')
        openAct.triggered.connect(self.openFileDialog)
        
        exitAct = QAction(QIcon('exit.png'), '&Exit', self)
        exitAct.setShortcut('Ctrl+Q')
        exitAct.setStatusTip('Exit application')
        exitAct.triggered.connect(qApp.quit)

        self.statusBar()

        menubar = self.menuBar()
        fileMenu = menubar.addMenu('&File')
        fileMenu.addAction(openAct)
        fileMenu.addAction(exitAct)

        self.img = QLabel(self)
        pixmap = QPixmap('00006.png')
        self.img.setPixmap(pixmap)

        self.text1 = QLabel('Upload an image for prediction')
        self.buttonO = QPushButton('Open')
        self.text2 = QLabel('Predicted Class : ')
        self.text3 = QLabel('None')

        self.buttonO.clicked.connect(self.openFileDialog)

        grid = QGridLayout()
        grid.setSpacing(10)

        grid.addWidget(self.text1, 1, 2, 1, 5)
        grid.addWidget(self.img, 3, 2, 5, 3)
        grid.addWidget(self.buttonO, 4, 5, 1, 2)
        grid.addWidget(self.text2, 5, 5, 1, 2)
        grid.addWidget(self.text3, 6, 5, 1, 2)
        
        widget = QWidget()
        widget.setLayout(grid)
        self.setCentralWidget(widget)

        self.setGeometry(100, 100, 600, 400)
        self.setWindowTitle('Traffic Sign Classifier')
        self.show()

    def openFileDialog(self):
        home_dir = str(os.getcwd())
        fname = QFileDialog.getOpenFileName(self, 'Open file', home_dir)
        pixmap = QPixmap(fname[0])
        self.img.setPixmap(pixmap.scaled(self.img.size()))
        image = Image.open(fname[0])
        image = image.resize((28, 28))
        image = np.expand_dims(image, axis=0)
        image = np.array(image)
        self.text3.setText(classes[model.predict_classes([image])[0]+1])



classes = { 
    1:'Speed limit (20km/h)',
    2:'Speed limit (30km/h)', 
    3:'Speed limit (50km/h)', 
    4:'Speed limit (60km/h)', 
    5:'Speed limit (70km/h)', 
    6:'Speed limit (80km/h)', 
    7:'End of speed limit (80km/h)', 
    8:'Speed limit (100km/h)', 
    9:'Speed limit (120km/h)', 
    10:'No passing', 
    11:'No passing vehicles over 3.5 tons', 
    12:'Right-of-way at intersection', 
    13:'Priority road', 
    14:'Yield', 
    15:'Stop', 
    16:'No vehicles', 
    17:'Vehicles > 3.5 tons prohibited', 
    18:'No entry', 
    19:'General caution', 
    20:'Dangerous curve left', 
    21:'Dangerous curve right', 
    22:'Double curve', 
    23:'Bumpy road', 
    24:'Slippery road', 
    25:'Road narrows on the right', 
    26:'Road work', 
    27:'Traffic signals', 
    28:'Pedestrians', 
    29:'Children crossing', 
    30:'Bicycles crossing', 
    31:'Beware of ice/snow',
    32:'Wild animals crossing', 
    33:'End speed + passing limits', 
    34:'Turn right ahead', 
    35:'Turn left ahead', 
    36:'Ahead only', 
    37:'Go straight or right', 
    38:'Go straight or left', 
    39:'Keep right', 
    40:'Keep left', 
    41:'Roundabout mandatory', 
    42:'End of no passing', 
    43:'End no passing vehicle > 3.5 tons' 
}

app = QApplication(sys.argv)
model = load_model('traffic_classifier_a.h5')
gui = tscGUI()
sys.exit(app.exec_())
