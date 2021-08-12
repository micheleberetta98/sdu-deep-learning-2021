# -*- coding: utf-8 -*-
"""
Created on Thu Aug 12 15:06:24 2021

@author: Admin
"""

import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import regularizers




# %% Reload and test old model

model = keras.models.load_model('../day-2/model-1-relu-relu-sigm.h5')

NUM_WORDS = 10000
(train_data, train_labels), (test_data, test_labels) = keras.datasets.imdb.load_data(num_words=NUM_WORDS)


def multi_hot_sequences(sequences, dimension):
    # Create an all-zero matrix of shape (len(sequences), dimension)
    results = np.zeros((len(sequences), dimension))
    for i, word_indices in enumerate(sequences):
        # set specific indices of results[i] to 1s
        results[i, word_indices] = 1.0
    return results


# %%
# train_data has shape (25_000, NUM_WORDS) ==> print(np.shape(train_data))
train_data = multi_hot_sequences(train_data, dimension=NUM_WORDS)
test_data = multi_hot_sequences(test_data, dimension=NUM_WORDS)

# %% Load your model again, and use model.evaluate to get the accuracy


score = model.evaluate(test_data, test_labels, verbose = 0) 

print('Test loss:', score[0]) 
print('Test accuracy:', score[1])

# %% Create new model

model2 = keras.Sequential()
model2.add(keras.layers.Dense(16, activation='relu', input_shape=(NUM_WORDS,), kernel_regularizer=regularizers.l1(1e-5)))
model2.add(keras.layers.Dense(16, activation='relu', kernel_regularizer=regularizers.l1(1e-5)))
model2.add(keras.layers.Dense(1, activation='sigmoid'))

model2.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# %% Train model
history = model2.fit(train_data, train_labels, epochs=5, validation_data=(test_data,test_labels))

# %%
score = model2.evaluate(test_data, test_labels, verbose = 0) 

print('Test loss:', score[0]) 
print('Test accuracy:', score[1])

# %% Create new model w/dropout

model3 = keras.Sequential()
model3.add(keras.layers.Dense(16, activation='relu', input_shape=(NUM_WORDS,), kernel_regularizer=regularizers.l2(1e-5)))
model3.add(keras.layers.Dense(16, activation='relu', kernel_regularizer=regularizers.l2(1e-5)))
model3.add(keras.layers.Dropout(0.7))
model3.add(keras.layers.Dense(1, activation='sigmoid'))


model3.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# %% Train model
history = model3.fit(train_data, train_labels, epochs=1, validation_data=(test_data,test_labels))

# %%
score = model3.evaluate(test_data, test_labels, verbose = 0) 

print('Test loss:', score[0]) 
print('Test accuracy:', score[1])

# %% Create new model4

model4= keras.Sequential()
model4.add(keras.layers.Dense(16, activation='relu', input_shape=(NUM_WORDS,), kernel_regularizer=regularizers.l2(1e-5), activity_regularizer=regularizers.l2(1e-5)))
model4.add(keras.layers.Dense(16, activation='relu', kernel_regularizer=regularizers.l2(1e-5), activity_regularizer=regularizers.l2(1e-5)))
# model4.add(keras.layers.Dropout(0.2))
model4.add(keras.layers.Dense(15, activation='relu', kernel_regularizer=regularizers.l2(1e-5), activity_regularizer=regularizers.l2(1e-5)))
model4.add(keras.layers.Dropout(0.7))
model4.add(keras.layers.Dense(1, activation='sigmoid'))


model4.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# %% Train model
history = model4.fit(train_data, train_labels, epochs=1, validation_data=(test_data,test_labels))

# %%
score = model4.evaluate(test_data, test_labels, verbose = 0) 

print('Test loss:', score[0]) 
print('Test accuracy:', score[1])



