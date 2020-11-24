import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import os
import time
import numpy as np
from lib_alexnet import get_label, get_images, create_model

test_labels = get_label("test")
test_images = get_images("test")

test_labels=np.expand_dims(test_labels, axis=1)

model=create_model()

checkpoint_path = "model_alexnet/cp.ckpt"

model.load_weights(checkpoint_path)
test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)
print("model, accuracy: {:5.2f}%".format(100 * test_acc))
