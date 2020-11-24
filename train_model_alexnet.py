import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import os
import time
import numpy as np
from lib_alexnet import get_label, get_images, create_model


CLASS_NAMES= ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10']

train_labels = get_label("train")
train_images = get_images("train")
validation_images = get_images("validation")
validation_labels = get_label("validation")

train_labels=np.expand_dims(train_labels, axis=1)
validation_labels=np.expand_dims(validation_labels, axis=1)


train_ds = tf.data.Dataset.from_tensor_slices((train_images, train_labels))
validation_ds = tf.data.Dataset.from_tensor_slices((validation_images, validation_labels))

def process_images(image, label):
    image = tf.image.per_image_standardization(image)
    image = tf.image.resize(image, (227,227))
    return image, label

train_ds_size = tf.data.experimental.cardinality(train_ds).numpy()
validation_ds_size = tf.data.experimental.cardinality(validation_ds).numpy()

train_ds = (train_ds
            .map(process_images)
            .shuffle(buffer_size=train_ds_size)
            .batch(batch_size=32, drop_remainder=True))
validation_ds = (validation_ds
            .map(process_images)
            .shuffle(buffer_size=train_ds_size)
            .batch(batch_size=32, drop_remainder=True))

model=create_model()


checkpoint_path = "model_alexnet/cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,save_weights_only=True,verbose=1)


model.fit(train_ds, epochs=30, validation_data=validation_ds, validation_freq=1, callbacks=[cp_callback])
test_loss, test_acc = model.evaluate(validation_images,  validation_labels, verbose=2)
print("model, accuracy: {:5.2f}%".format(100 * test_acc))