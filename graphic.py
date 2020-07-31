from PyQt5.QtWidgets import (qApp, QLabel, QFrame, QAction, QWidget, QTextEdit, QGridLayout,
                             QPushButton, QMainWindow, QApplication, QFileDialog)
from PyQt5.QtGui import QIcon, QPixmap
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
        image = image.resize((30, 30))
        image = np.expand_dims(image, axis=0)
        image = np.array(image)
        self.text3.setText(classes[model.predict_classes([image])[0]])

classes = ['20 km/h', '30 km/h', '50 km/h', '60 km/h', '70 km/h', '80 km/h', '80 km/h end', 
           '100 km/h', '120 km/h', 'No overtaking', 'No overtaking for tracks', 'Crossroad with secondary way', 
           'Main road', 'Give way', 'Stop', 'Road up', 'Road up for track', 'Brock', 'Other dangerous', 
           'Turn left', 'Turn right', 'Winding road', 'Hollow road', 'Slippery road', 'Narrowing road', 
           'Roadwork', 'Traffic light', 'Pedestrian', 'Children', 'Bike', 'Snow', 'Deer', 'End of the limits', 
           'Only right', 'Only left', 'Only straight', 'Only straight and right', 'Only straight and left', 
           'Take right', 'Take left', 'Circle crossroad', 'End of overtaking limit', 'End of overtaking limit for track']

app = QApplication(sys.argv)
model = load_model('traffic_classifier.h5')
gui = tscGUI()
sys.exit(app.exec_())

