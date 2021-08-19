# -*- coding: utf-8 -*-
"""
Task 4 - Visualizing your results
Finally, you must visualize some aspects of your model. It can be a graph of the
training/validation performance, visualization of the 
lters or feature maps, oranything you can think of. This has to be saved as an image 
le, and uploaded along with your model and code.

"""
#   Imports
import seaborn as sns
from keras_visualizer import visualizer 

%matplotlib inline
from sklearn.metrics import confusion_matrix
import itertools
import matplotlib.pyplot as plt

from collections import OrderedDict
import numpy as np
from keras.datasets import mnist
from matplotlib import pyplot as plt
from tensorflow.keras.layers import Conv2D, Input
from tensorflow.keras.models import Model
from tensorflow.python.keras.models import load_model

#%% Visualize Keras Models
# !pip3 install keras-visualizer

def visualize_model():
    return visualizer(model, format='png', view=True)

#%% Visualize Batch

def show_batch(image_batch, label_batch):
    plt.figure(figsize=(10,10))
    for n in range(25):
        ax = plt.subplot(5,5,n+1)
        plt.imshow(image_batch[n])
        if label_batch[n]:
            plt.title("PNEUMONIA")
        else:
            plt.title("NORMAL")
        plt.axis("off")
    return plt.show()
        
#%% Plot Loss and Accuracy

def show_plot_loss(subfig, h):
    loss_values = h.history['loss']
    val_loss_values = h.history['val_loss']
    epochs = range(1, len(loss_values) + 1)

    subfig.plot(epochs, loss_values, label='Training loss')
    subfig.plot(epochs, val_loss_values, label='Validation loss')
    subfig.set(xlabel='Epochs', ylabel='Loss')
    subfig.legend()


def show_plot_accuracy(subfig, h):
    accuracy_values = h.history['accuracy']
    val_accuracy_values = h.history['val_accuracy']
    epochs = range(1, len(accuracy_values) + 1)

    subfig.plot(epochs, accuracy_values, label='Training accuracy')
    subfig.plot(epochs, val_accuracy_values, label='Validation accuracy')
    subfig.set(xlabel='Epochs', ylabel='Accuracy')
    subfig.legend()
    
    
#%% Confusion Matrix

def show_confusion_matrix(y_true, y_pred):
    
    # Create confusion matrix
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
    
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            print("Normalized confusion matrix")
        else:
            print('Confusion matrix, without normalization')
    
        print(cm)
    
        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, cm[i, j],
                horizontalalignment="center",
                color="white" if cm[i, j] > thresh else "black")
    
        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        
    # Define labels for the confusion matrix
    cm_plot_labels = ['pneunomia','normal']
    
    plot_confusion_matrix(cm=cm, classes=cm_plot_labels, title='Confusion Matrix')

#%% Show several images

def show_several_images(X_train):
    fig, axes = plt.subplots(ncols=4, nrows=4, figsize=(7, 7))
    
    indices = np.random.choice(len(X_train), 14)
    counter = 0
    
    for i in range(2):
        for j in range(7):
            axes[i,j].set_title(y_train[indices[counter]])
            axes[i,j].imshow(X_train[indices[counter]], cmap='gray')
            axes[i,j].get_xaxis().set_visible(False)
            axes[i,j].get_yaxis().set_visible(False)
            counter += 1
    plt.show()

#%% Feature Map

def show_feature_map():
        
    def get_number_indexes(labels):
        indx = {}
        for i, f in enumerate(labels):
            if len(indx) == 10:
                break
            if f not in indx:
                indx[f] = i
        return list(OrderedDict(sorted(indx.items())).values())
    
    
    inputs = Input((28, 28, 1))
    x = Conv2D(32, (3, 3), activation='relu')(inputs)
    model = Model(inputs, x)
    old_model_weights = load_model('./model.h5').layers[1].get_weights()  # <- Be sure to point to the correct model and conv layer
    model.layers[1].set_weights(old_model_weights)
    for f in get_number_indexes(test_labels):
        feature_map_data = test_data[f].reshape(1, 28, 28, 1)
        feature_maps = model.predict(feature_map_data)
        square = 5
        for ix in range(np.square(square)):
            ax = plt.subplot(square, square, ix+1)
            ax.set_xticks([])
            ax.set_yticks([])
            plt.imshow(feature_maps[0, :, :, ix], cmap='gray')
        plt.show()
    
# %% Activations

# layer_outputs = [layer.output for layer in model.layers]
# activation_model = models.Model(inputs=model.input, outputs=layer_outputs)
# activations = activation_model.predict(train_data[10].reshape(1, 28, 28, 1))


def display_activation(activations, row_size, col_size, act_index):
    activation = activations[act_index]
    activation_index = 0
    fig, ax = plt.subplots(
        row_size, col_size, figsize=(row_size*2.5, col_size*1.5))

    for row in range(0, row_size):
        for col in range(0, col_size):
            ax[row][col].imshow(
                activation[0, :, :, activation_index], cmap='gray')
            activation_index += 1

