from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
import matplotlib.pyplot as plt
import tensorflow as tf
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
            batch_size=32,
            subset='training')
    
    validation_generator = train_datagen.flow_from_directory(
            directory=train_dir,
            target_size=(IMG_HEIGHT, IMG_WIDTH),
            shuffle=True,
            batch_size=32,
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
EPOCHS = 50

train_gen, val_gen = load_imgs()
sample_training_images, _ = next(train_gen)

Adam = tf.keras.optimizers.Adam(learning_rate=1e-3)
early_stopping1 = EarlyStopping(monitor='loss', mode='min', verbose=2, patience=5)
early_stopping2 = EarlyStopping(monitor='val_loss', mode='min', verbose=2, patience=10)
best_model1 = ModelCheckpoint('traffic_classifier_v.h5', monitor='val_accuracy', mode='max', verbose=1, save_best_only=True)
best_model2 = ModelCheckpoint('traffic_classifier_a.h5', monitor='accuracy', mode='max', verbose=1, save_best_only=True)

model = Sequential()
model.add(tf.keras.layers.Conv2D(16, (3, 3), activation='relu', input_shape=(IMG_WIDTH, IMG_HEIGHT, 3)))
model.add(tf.keras.layers.MaxPooling2D((2, 2)))
model.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal'))
model.add(tf.keras.layers.MaxPooling2D((2, 2)))
model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal'))
model.add(tf.keras.layers.MaxPooling2D((2, 2)))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(1024, activation='relu'))
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(256, activation='relu'))
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(128, activation='relu'))
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(43, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer=Adam, metrics=['accuracy'])
history = model.fit(train_gen, 
                    epochs=EPOCHS, 
                    validation_data=val_gen, 
                    callbacks=[early_stopping1, 
                               early_stopping2, 
                               best_model1, 
                               best_model2]
                    )

plot_train_charts(history)

