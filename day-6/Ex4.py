# -*- coding: utf-8 -*-
"""
Deep Learning Summer Course - Exercise 4

"""

from tensorflow.keras.datasets import mnist
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import optimizers
from tensorflow.keras import models
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt


# %%

(train_data, train_labels), (test_data, test_labels) = mnist.load_data()

# %%

train_data = train_data.reshape((len(train_data), 28, 28, 1)) / 255
test_data = test_data.reshape((len(test_data), 28, 28, 1)) / 255

train_labels = to_categorical(train_labels, num_classes=10)
test_labels = to_categorical(test_labels, num_classes=10)


# %%

seaborn.heatmap(train_data[1000, :, :].reshape((28,28)))

# %%

model = models.Sequential()

model.add(layers.Conv2D(32, (5, 5), activation='relu', input_shape=(28, 28, 1), data_format='channels_last', name='first_layer',))
model.add(layers.MaxPooling2D((2, 2)))
# model.add(layers.Conv2D(64, (3, 3), activation='relu', data_format='channels_last'))
# model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu', data_format='channels_last'))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))

# %%

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# %%

history = model.fit(train_data, train_labels, epochs=2, validation_data=(test_data,test_labels))


# %%
kernels = model.get_layer(name='first_layer').get_weights()[0][:, :, 0, :]
sns.heatmap(kernels[:, :, 1])

# %%

layer_outputs= [layer.output for layer in model.layers]
activation_model= models.Model(inputs=model.input, outputs=layer_outputs)
activations = activation_model.predict(train_data[10].reshape(1,28,28,1))

def display_activation(activations, col_s, row_s, act_index):
    activation = activations[act_index]
    activation_index=0
    fig, ax = plt.subplots(row_s, col_s, figsize=(row_s*2.5,col_s*1.5))

    for row in range(0,row_s):
        for col in range(0,col_s):
            ax[row][col].imshow(activation[0, :, :, activation_index], cmap='gray')
            activation_index+= 1
        