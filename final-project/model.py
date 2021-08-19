'''
This file contains the final model which has been tested and tweaked
in order to have the highest accuracy possible

@author Michele Beretta (model testing)
@author Bianca Crippa   (model testing)
@author Tanja Gurtner   (visualizations)
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

# %% Visualize Activations
from tensorflow.keras.models import Model
import matplotlib.pyplot as plt

layer_outputs= [layer.output for layer in model.layers]
activation_model= Model(inputs=model.input, outputs=layer_outputs)

def display_activation(activations, col_size, row_size, act_index):
    activation = activations[act_index]
    activation_index=0
    fig, ax = plt.subplots(row_size, col_size, figsize=(row_size*2.5,col_size*1.5))
    for row in range(0,row_size):
        for col in range(0,col_size):
            ax[row][col].imshow(activation[activation_index, :, :, 0], cmap='gray')
            activation_index+= 1


activations = activation_model.predict(test_generator[3])
display_activation(activations, 9, 5, 0)   
    
    
#%% Confusion Matrix
from sklearn.metrics import confusion_matrix
import itertools
import numpy as np

# print(train_generator.class_indices)

probabilities = model.predict_generator(generator=test_generator)
y_true = test_generator.classes
y_pred = probabilities > 0.5

cm = confusion_matrix(y_true, y_pred)

# Function provided by https://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html#sphx-glr-auto-examples-model-selection-plot-confusion-matrix-py
def plot_confusion_matrix(cm, classes,
                        normalize=False,
                        title='Confusion matrix',
                        cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
            horizontalalignment="center",
            color="red")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig('confusion')


cm_plot_labels = ['pneunomia','normal']

plot_confusion_matrix(cm=cm, classes=cm_plot_labels, title='Confusion Matrix')


#%% Kernel Heatmap

import seaborn

layer_names = []
for layer in model.layers:
    #  if layer.name.startswith("conv"):
    layer_names.append(layer.name)


for layer in layer_names:
    try:
        kernels = model.get_layer(name=layer).get_weights()[0][:, :, 0, :] 
        
        fig = plt.figure(figsize=(12,12))
        plt.suptitle("{}: Kernels Heatmap".format(layer))
        
        for i in range(n_filters):
            ax = fig.add_subplot(6,6, i+1)
            imgplot = seaborn.heatmap(kernels[:,:,i])
            ax.set_title('Kernel No. {}'.format(i))
            ax.set_aspect('equal', adjustable='datalim')
                   
        fig.tight_layout()
        plt.savefig('kernel_heatmap_{}'.format(layer))
        plt.show()
    except IndexError:
        print("Not possible for layer: ",layer)
    except:
        print("Error for layer: ",layer)
