from keras.layers import Conv2D, MaxPool2D, Dense, Flatten, Dropout
from keras.utils import to_categorical
from keras.models import Sequential
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import tensorflow as tf
from PIL import Image
import pandas as pd 
import numpy as np 
import cv2
import os


def load_images():
    data = []
    labels = []

    for i in range(classes) :
        path = "./train/{0}/".format(i)

        for img in os.listdir(path):
            try:
                image = cv2.imread(path+img)
                image = Image.fromarray(image, 'RGB')
                image = image.resize((height, width))
                data.append(np.array(image))
                labels.append(i)
            except AttributeError:
                print(" ")
                
    data = np.array(data)
    labels = np.array(labels)

    print('Images Loaded')
    return data, labels

def preprocess_data(X, y):
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2)

    y_train = to_categorical(y_train, 43)
    y_valid = to_categorical(y_valid, 43)

    X_train = X_train.astype('float32') / 255 
    X_valid = X_valid.astype('float32') / 255

    print('Processed Data')
    return X_train, X_valid, y_train, y_valid

def calculate_weights():
    no_obj = []
    paths = []
    
    train_dir = os.path.join(root, 'Train')

    for i in range(classes):
        curr = "./train/{0}/".format(i)
        no_obj.append(len(os.listdir(curr)))
        paths.append(int(i))

    no_obj = np.array(no_obj)
    weights = np.max(no_obj) / no_obj

    return dict(zip(paths, weights))

def plot_train_charts(history):
    fig = plt.subplots(figsize=(10, 4))

    plt.subplot(121)
    plt.title('Accuracy')
    plt.plot(history.history['accuracy'], label='Train')
    plt.plot(history.history['val_accuracy'], label='Valid')
    plt.legend()
    
    plt.subplot(122)
    plt.title('Loss')
    plt.plot(history.history['loss'], label='Train')
    plt.plot(history.history['val_loss'], label='Valid')
    plt.legend()
    plt.show()

height = 30
width = 30
channels = 3
classes = 43
epochs = 20

X, y = load_images()
X_train, X_valid, y_train, y_valid = preprocess_data(X, y)

weights = calculate_weights()

model = Sequential()
model.add(Conv2D(filters=16, kernel_size=(3, 3), activation='relu', input_shape=X_train.shape[1:]))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu'))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(1024, activation='relu'))
model.add(Dropout(rate=0.5))
model.add(Dense(43, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

history = model.fit(X_train, y_train, 
                    batch_size=32, 
                    epochs=epochs, 
                    class_weight=weights, 
                    validation_data=(X_valid, y_valid))

plot_train_charts(history)
model.save('traffic_classifier.h5')
