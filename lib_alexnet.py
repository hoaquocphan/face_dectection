from os import walk
import os
# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras
# Helper libraries
import numpy as np
import matplotlib.pyplot as plt
import cv2

def preprocess_image(img):   
    image = cv2.imread(img)
    input_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
    input_img = cv2.resize(input_img, (227, 227)) 
    return input_img

def get_label(dir):
    image_name = []
    labels = []
    for (dirpath, dirnames, filenames) in walk(dir):
        image_name.extend(filenames)
    for image in image_name:
        labels.append(image.split("_")[0])
    labels = [int(numeric_string)-1 for numeric_string in labels]
    labels=np.array(labels)
    return labels

def get_images(dir):
    image_name=[]
    images =[]
    for (dirpath, dirnames, filenames) in walk(dir):
        image_name.extend(filenames)
    for image in image_name:
        image_file = dir + "/" + image
        image=preprocess_image(image_file)
        images.append(image)
    images = np.array(images)
    return images


def create_model():
    model = keras.models.Sequential([
    keras.layers.Conv2D(filters=96, kernel_size=(11,11), strides=(4,4), activation='relu', input_shape=(227,227,3)),
    keras.layers.BatchNormalization(),
    keras.layers.MaxPool2D(pool_size=(3,3), strides=(2,2)),
    keras.layers.Conv2D(filters=256, kernel_size=(5,5), strides=(1,1), activation='relu', padding="same"),
    keras.layers.BatchNormalization(),
    keras.layers.MaxPool2D(pool_size=(3,3), strides=(2,2)),
    keras.layers.Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), activation='relu', padding="same"),
    keras.layers.BatchNormalization(),
    keras.layers.Conv2D(filters=384, kernel_size=(1,1), strides=(1,1), activation='relu', padding="same"),
    keras.layers.BatchNormalization(),
    keras.layers.Conv2D(filters=256, kernel_size=(1,1), strides=(1,1), activation='relu', padding="same"),
    keras.layers.BatchNormalization(),
    keras.layers.MaxPool2D(pool_size=(3,3), strides=(2,2)),
    keras.layers.Flatten(),
    keras.layers.Dense(4096, activation='relu'),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(4096, activation='relu'),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(10, activation='softmax')
    ])
    model.compile(loss='sparse_categorical_crossentropy', optimizer=tf.optimizers.SGD(lr=0.001), metrics=['accuracy'])
    model.summary()
    return model
