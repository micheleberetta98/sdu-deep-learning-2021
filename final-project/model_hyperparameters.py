'''
    This file is used to test all the different models and then take the best one
    @author Michele Beretta
    @author Bianca Crippa
'''

# %% Imports

import tensorflow as tf
import wandb
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Input
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping
from wandb.keras import WandbCallback

from constants import IMG_SIZE, STEPS_PER_EPOCH, VAL_STEPS
from image_generation import train_generator, test_generator, validation_generator

# %% Function utilities for creating and validating models


def create_model(name='', num_of_kernels=32, kernel_size=(5, 5)):
    '''A function with some useful parameters in order to create easily different models'''

    inputs = Input(shape=(IMG_SIZE, IMG_SIZE, 1), name='input_layer')
    x = inputs

    # We work on images, so a CNN (Convolutional Neural Network) is useful
    # Here we try a couple of convolutional layers with max pooling
    x = Conv2D(num_of_kernels, kernel_size, activation='relu', name='conv_2d_1')(x)
    x = MaxPooling2D((3, 3), name='max_2d_1')(x)
    x = Conv2D(num_of_kernels, kernel_size, activation='relu', name='conv_2d_2')(x)
    x = MaxPooling2D((3, 3), name='max_2d_2')(x)

    x = Flatten(name='flatten')(x)

    # After the convolutional, a couple of dense layers to learn
    x = Dense(64, activation='relu', name='dense_1')(x)
    x = Dense(64, activation='relu', name='dense_2')(x)
    x = Dense(64, activation='relu', name='dense_3')(x)

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
    '''Utility function to evaluate a model on a given validation generator'''
    score = model.evaluate(val_generator, verbose=verbose)
    return score[0], score[1]


# Early stopping can be useful in order to prevent excessive training
# and stopping at the best model if loss doesn't get better
early_stopping = EarlyStopping(
    patience=3,
    monitor="val_loss",
    restore_best_weights=True,
)


def fit_model(model):
    '''Utility function to fit the model with the correct callbacks (early stopping and wandb)'''
    wandb.init(project='sdu-deep-learning-final', entity='micheleberetta98', name=model.name)
    wandb_callback = WandbCallback(log_weights=True,
                                   log_evaluation=True,
                                   save_model=True,
                                   validation_steps=VAL_STEPS)
    history = model.fit(train_generator,
                        steps_per_epoch=STEPS_PER_EPOCH,
                        validation_steps=VAL_STEPS,
                        validation_data=validation_generator,
                        epochs=10,
                        use_multiprocessing=True,
                        workers=8,
                        verbose=2,
                        callbacks=[early_stopping, wandb_callback])
    wandb.finish()

    return history


def test_model(ks=[], create=lambda x: x):
    '''Utility functions that goes through all hyperparameters ks and choose the one that minimizes test loss'''
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

# Here we test different hyperparameters for the convolutional part:
# - the kernel size
# - the number of filters


if __name__ == '__main__':
    kernel, model_kernel, history_kernel = test_model(
        ks=[(3, 3), (5, 5), (7, 7)],
        create=lambda x: create_model(name=f'model_kernel_{x[0]}x{x[1]}', kernel_size=x))

# The best we have found are 5x5 and 7x7 (they seems to behave similarly)

# %% Finding the best number of filters

if __name__ == '__main__':
    base_name = f'model_k7x7'
    num_of_filters, model_num_filters, history_num_filters = test_model(
        ks=[8, 16, 32],
        create=lambda x: create_model(name=f'{base_name}_num_kernels_{x}', kernel_size=(7, 7))
    )

# The best we have found is 32
