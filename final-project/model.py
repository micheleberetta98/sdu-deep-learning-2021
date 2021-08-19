# %% Imports

import tensorflow as tf
import wandb
from wandb.keras import WandbCallback
from tensorflow.keras.layers import Dense, Conv2D, Concatenate, AvgPool2D, MaxPooling2D, Flatten, Dropout, Input
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping

from constants import IMG_SIZE, NUM_OF_IMAGES_VAL, NUM_OF_IMAGES_TRAIN, EPOCHS, BATCH_SIZE
from image_generation import validation_generator, train_generator, test_generator
from model_hyperparameters import evaluate_model

# %% Model definition

# The hyperparameters
kernel_size = (7, 7)
n_filters = 32
input_dropout = 0
dense_dropout = 0
units = [128, 128, 128]

# This is the final model comprising of all of the hyperparameters seen in model_hyperparameters
# Here we test for dropout and regularization in order to reduce overfitting

inputs = Input(shape=(IMG_SIZE, IMG_SIZE, 1), name='input_layer')
x = inputs

# Dropout after the input could be beneficial in order to generalize better
# and reduce overfitting
x = Dropout(input_dropout, name='input_dropout')(x)

# We work on images, so a CNN (Convolutional Neural Network) is useful
# Here we try a couple of convolutional layers with max pooling 

x = Conv2D(n_filters, kernel_size, activation='relu', name='conv_2d_1')(x)
x = MaxPooling2D((3, 3), name='max_2d_1')(x)
x = Conv2D(n_filters, kernel_size, activation='relu', name='conv_2d_2')(x)
x = MaxPooling2D((3, 3), name='max_2d_2')(x)
x = Flatten(name='flatten_1')(x)

# After the convolutional, a couple of dense layers to learn
x = Dense(units[0], activation='relu', name='dense_1')(x)
x = Dense(units[1], activation='relu', name='dense_2')(x)
x = Dense(units[2], activation='relu', name='dense_3')(x)

# Since we are dealing with a binary problem (pneumonia or normal),
# the output is given by this 1 neuron with a sigmoid activation function
output = Dense(1, activation='sigmoid', name='dense_output')(x)

model = Model(inputs=inputs, outputs=output, name='model_2')

# Binary crossentropy as a loss function is ideal for a binary problem
model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# %% Fitting process

steps_per_epoch = NUM_OF_IMAGES_TRAIN // BATCH_SIZE
val_steps = NUM_OF_IMAGES_VAL // BATCH_SIZE

# Early stopping can be useful in order to prevent excessive training
# and stopping at the best model if accuracy doesn't get better
early_stopping = EarlyStopping(
    patience=3,
    monitor='val_accuracy',
    mode='max',
    restore_best_weights=True,
)

wandb.init(project='sdu-deep-learning-final', entity='micheleberetta98', name=model.name)
wandb_callback = WandbCallback(log_weights=True,
                                log_evaluation=True,
                                save_model=True,
                                validation_steps=val_steps)
history = model.fit(train_generator,
                    steps_per_epoch=steps_per_epoch,
                    validation_steps=val_steps,
                    validation_data=validation_generator,
                    epochs=EPOCHS,
                    use_multiprocessing=True,
                    workers=8,
                    callbacks=[early_stopping, wandb_callback])
loss, acc = evaluate_model(model, test_generator)
print(f'Loss = {loss}')
print(f'Acc  = {acc}')

# %%

model.save(f'model_2_dropout_0.h5')

# %%
