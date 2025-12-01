import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
(x_train_image, y_train_label), (x_test_image, y_test_label) = tf.keras.datasets.cifar10.load_data()
x_train_normalized = x_train_image.astype('float32') / 255.0
x_test_normalized = x_test_image.astype('float32') / 255.0
y_train = y_train_label.flatten()  # 从(50000,1)变为(50000,)
y_test = y_test_label.flatten()
model = keras.Sequential()
model.add(keras.layers.Conv2D(filters=6, kernel_size=3, strides=(1,1),input_shape=(32,32,3)))
model.add(keras.layers.MaxPooling2D(pool_size=(2,2),strides=2))
model.add(keras.layers.ReLU())
model.add(keras.layers.Conv2D(filters=16, kernel_size=3, strides=(1,1)))
model.add(keras.layers.MaxPooling2D(pool_size=(2,2),strides=2))
model.add(keras.layers.ReLU())
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(units=120, activation='relu'))
model.add(keras.layers.Dense(units=84, activation='relu'))
model.add(keras.layers.Dense(units=10, activation='softmax'))
model.summary()
model.compile(loss='sparse_categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])
model_train = model.fit(x_train_normalized, y_train,validation_split=0.2, epochs=10, batch_size=300, verbose=1)