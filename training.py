import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
import matplotlib.pyplot as plt
import numpy as np
import os


def load_imgs():
    root = './'
    train_dir = os.path.join(root, 'Train')
    train_datagen = ImageDataGenerator(rescale=1./255, 
                                       brightness_range=(0.8, 1.1), 
                                       rotation_range=10,
                                       width_shift_range=0.1,
                                       height_shift_range=0.1,
                                       shear_range=0.15,
                                       horizontal_flip=False,
                                       vertical_flip=False,
                                       fill_mode='nearest',
                                       validation_split=0.1)

    train_generator = train_datagen.flow_from_directory(
            directory=train_dir,
            target_size=(IMG_HEIGHT, IMG_WIDTH),
            shuffle=True,
            batch_size=128,
            subset='training')
    
    validation_generator = train_datagen.flow_from_directory(
            directory=train_dir,
            target_size=(IMG_HEIGHT, IMG_WIDTH),
            shuffle=True,
            batch_size=128,
            subset='validation')

    return train_generator, validation_generator

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


IMG_HEIGHT = 28
IMG_WIDTH = 28
EPOCHS = 30

train_gen, val_gen = load_imgs()
sample_training_images, _ = next(train_gen)

Adam = tf.keras.optimizers.Adam(learning_rate=0.005)

model = Sequential([
    tf.keras.layers.Conv2D(16, 3, activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
    tf.keras.layers.Dropout(0.25),
    tf.keras.layers.MaxPool2D(),
    tf.keras.layers.Conv2D(32, 3, activation='relu'),
    tf.keras.layers.Dropout(0.25),
    tf.keras.layers.MaxPool2D(),
    tf.keras.layers.Conv2D(64, 3, activation='relu'),
    tf.keras.layers.Dropout(0.25),
    tf.keras.layers.MaxPool2D(),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dropout(0.25),
    tf.keras.layers.Dense(43, activation='softmax')
])

model.compile(loss='categorical_crossentropy', optimizer=Adam, metrics=['accuracy'])
history = model.fit(train_gen, epochs=EPOCHS, validation_data=val_gen)

plot_train_charts(history)

model.save('traffic_classifier.h5')

