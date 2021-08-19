'''
    This file is used to test all the different models and then take the best one
    @author Michele Beretta
    @author Bianca Crippa
'''

# %% Imports

import tensorflow as tf
import wandb
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout, Input
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping
from wandb.keras import WandbCallback

from constants import IMG_SIZE, NUM_OF_IMAGES_TRAIN, NUM_OF_IMAGES_VAL, BATCH_SIZE, EPOCHS
from image_generation import train_generator, test_generator, validation_generator

# %% Function utilities for creating and validating models


def create_model(name='', input_dropout=0, num_of_kernels=32, dense_dropout=0, kernel_size=(5, 5), dense_units=64):
    '''A function with some useful parameters in order to create easily different models'''
    inputs = Input(shape=(IMG_SIZE, IMG_SIZE, 1), name='input_layer')
    x = inputs
    x = Dropout(input_dropout, name='input_dropout')(x)

    # We work on images, so a CNN (Convolutional Neural Network) is useful
    # Here we try a couple of convolutional layers with max pooling
    x = Conv2D(num_of_kernels, kernel_size, activation='relu', name='conv_2d_1')(x)
    x = MaxPooling2D((3, 3), name='max_2d_1')(x)
    x = Conv2D(num_of_kernels, kernel_size, activation='relu', name='conv_2d_2')(x)
    x = MaxPooling2D((3, 3), name='max_2d_2')(x)

    x = Flatten(name='flatten')(x)

    # After the convolutional, a couple of dense layers to learn
    # This number of layers could be cross-val-decided too, as well as the number of units
    # Also maybe adding some dropout in the middle?
    x = Dense(dense_units, activation='relu', name='dense_1')(x)
    x = Dropout(dense_dropout, name='dense_dropout1')(x)
    x = Dense(dense_units, activation='relu', name='dense_2')(x)
    x = Dropout(dense_dropout, name='dense_dropout2')(x)
    x = Dense(dense_units, activation='relu', name='dense_3')(x)

    # Since we are dealing with a binary problem (pneumonia or normal),
    # the output is given by this 1 neuron with a sigmoid activation function
    output = Dense(1, activation='sigmoid', name='dense_output')(x)

    model = Model(inputs=inputs, outputs=output, name=name)

    model.compile(optimizer='adam',
                  # Binary crossentropy as a loss function is ideal for a binary problem
                  loss='binary_crossentropy',
                  # We are interested in the network accuracy (and also the loss)
                  metrics=['accuracy'])
    return model


def evaluate_model(model, val_generator, verbose=0):
    score = model.evaluate(val_generator, verbose=verbose)
    return score[0], score[1]

# We calculate the steps per epoch so that we use all the images
# we have in the folders
steps_per_epoch = NUM_OF_IMAGES_TRAIN // BATCH_SIZE
val_steps = NUM_OF_IMAGES_VAL // BATCH_SIZE

# Early stopping can be useful in order to prevent excessive training
# and stopping at the best model if loss doesn't get better
early_stopping = EarlyStopping(
    patience=3,
    monitor="val_loss",
    restore_best_weights=True,
)

def fit_model(model):
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
                        verbose=2,
                        callbacks=[early_stopping, wandb_callback])

    return history

def test_model(ks=[], create=lambda x: x):
    best_model = None
    best_k = None
    best_loss = float('inf')
    best_history = None

    # Here we test against all the hyperparameters in ks
    # and keep only the model that with the best val_loss
    # for a specific hyperparameter
    for k in ks:
        print(f'Model with {k} ... ', end='')
        model = create(k)
        history = fit_model(model)
        loss, acc = evaluate_model(model, test_generator)
        print(f"Loss = {loss}")
        print(f"Acc  = {acc}")
        if loss < best_loss:
            best_k = k
            best_loss = loss
            best_model = model
            best_history = history

    return best_k, best_model, best_history

# %% Finding the best kernel size

kernel, model_kernel, history_kernel = test_model(
    ks=[(3, 3), (5, 5), (7, 7)],
    create=lambda x: create_model(name=f'model_kernel_{x[0]}x{x[1]}', kernel_size=x))

# %% Finding the best number of kernels

base_name = f'model_k{kernel[0]}x{kernel[1]}'
num_of_kernels, model_num_kernels, history_num_kernels = test_model(
    ks=[8, 16, 32],
    create=lambda x: create_model(name=f'{base_name}_num_kernels_{x}', kernel_size=kernel)
)

# %% Finding the best units for the dense layer

units, model_units, history_units = test_model(
    ks=[16, 32, 64, 128],
    create=lambda x: create_model(name=f'model_units_{x}',
                                  kernel_size=kernel,
                                  num_of_kernels=num_of_kernels,
                                  dense_units=x)
)

# %% Putting together and training the model

model = create_model(name='model_no_reg',
                     num_of_kernels=num_of_kernels,
                     kernel_size=kernel,
                     dense_units=units)

history = fit_model(model)
loss, acc = evaluate_model(model, validation_generator)
print(f'Loss = {loss}')
print(f'Acc  = {acc}')

# %% Adding dropout

input_drop, _, _ = test_model(
    ks=[0.1, 0.2, 0.4, 0.5, 0.6],
    create=lambda x: create_model(
        name=f'model_in_drop_{x}',
        input_dropout=x,
        num_of_kernels=num_of_kernels,
        kernel_size=kernel,
        dense_units=units)
)

# %% Save the model

model.save('model.h5')