'''
This file contains the final model which has been tested and tweaked
in order to have the highest accuracy possible

@author Michele Beretta
@author Bianca Crippa
'''

# %% Imports

import tensorflow as tf
import wandb
from wandb.keras import WandbCallback
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout, Input
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.regularizers import l2
import matplotlib.pyplot as plt

from constants import IMG_SIZE, NUM_OF_IMAGES_VAL, NUM_OF_IMAGES_TRAIN, BATCH_SIZE
from image_generation import validation_generator, train_generator, test_generator
from model_hyperparameters import evaluate_model

# %% Model definition

# The hyperparameters
kernel_size = (5, 5)
n_filters = 32
dense_dropout = 0.1
units = [128, 128, 64]
l2_k = 0.01

# Kernel size and the number of convolutional layers were chosen as explained in model_hyperparameters.py
# The other ones were chosen by trial and error.
# We also noticed that a dropout right after the input wasn't so beneficial as an output in the middle of the
# model, and that a l1 regularizer (or a l1-l2) was too much aggressive so we went for a l2 regularizer.
# Moreover, we tried to have multiple branches on the convolutional aprt, but performance didn't seem
# to be affected a lot positively, so we went for a simpler linear model.

# This is the final model comprising of all of the hyperparameters seen in model_hyperparameters
# Here we test for dropout and regularization in order to reduce overfitting

inputs = Input(shape=(IMG_SIZE, IMG_SIZE, 1), name='input_layer')
x = inputs

# We work on images, so a CNN (Convolutional Neural Network) is useful
# Here we try a couple of convolutional layers with max pooling 

x = Conv2D(n_filters, kernel_size, activation='relu', name='conv_2d_1')(x)
x = MaxPooling2D((3, 3), name='max_2d_1')(x)
x = Conv2D(n_filters, kernel_size, activation='relu', name='conv_2d_2')(x)
x = MaxPooling2D((3, 3), name='max_2d_2')(x)

# Dropout after the convolutional layers could be beneficial in order to generalize better
# and reduce overfitting on the dense layers
x = Dropout(dense_dropout, name='dense_dropout')(x)
x = Flatten(name='flatten_1')(x)

# After the convolutional, a couple of dense layers to learn
x = Dense(units[0], activation='relu', kernel_regularizer=l2(l2_k), name='dense_1')(x)
x = Dense(units[1], activation='relu', kernel_regularizer=l2(l2_k), name='dense_2')(x)
x = Dense(units[2], activation='relu', kernel_regularizer=l2(l2_k), name='dense_3')(x)

# Since we are dealing with a binary problem (pneumonia or normal),
# the output is given by this 1 neuron with a sigmoid activation function
output = Dense(1, activation='sigmoid', name='dense_output')(x)

model = Model(inputs=inputs, outputs=output, name='model_2_dropout_reg-l2-0.01_middle01')

# Binary crossentropy as a loss function is ideal for a binary problem
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# %% Fitting process

steps_per_epoch = NUM_OF_IMAGES_TRAIN // BATCH_SIZE
val_steps = NUM_OF_IMAGES_VAL // BATCH_SIZE

# Early stopping can be useful in order to prevent excessive training
# and stopping at the best model if accuracy doesn't get better
early_stopping = EarlyStopping(
    patience=7,
    monitor='val_accuracy',
    mode='max',
    restore_best_weights=True,
)

# We are using wandb to log the data and to provide graphs, which are visible
# at https://wandb.ai/micheleberetta98/sdu-deep-learning-final?workspace=user-micheleberetta98
# (we also saved the images for the best networks)
wandb.init(project='sdu-deep-learning-final', entity='micheleberetta98', name=model.name)
wandb_callback = WandbCallback(log_weights=True,
                                log_evaluation=True,
                                save_model=True,
                                validation_steps=val_steps)
history = model.fit(train_generator,
                    steps_per_epoch=steps_per_epoch,
                    validation_steps=val_steps,
                    validation_data=validation_generator,
                    epochs=15,
                    use_multiprocessing=True,
                    workers=8,
                    callbacks=[early_stopping, wandb_callback])
wandb.finish()
h = history.history
print(h)
loss, acc = evaluate_model(model, test_generator)
print(f'Loss = {loss}')
print(f'Acc  = {acc}')

# %% Saving the model for future use

model.save(f'{model.name}.h5')

# %%
