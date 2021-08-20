# -*- coding: utf-8 -*-
"""
Visualizing
Created on Thu Aug 19 21:35:19 2021

@author: Admin
"""
 
 # %%

from image_generation import test_generator
from tensorflow.keras.models import Model, load_model
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model
from sklearn.metrics import confusion_matrix
import itertools
import numpy as np

import seaborn


model_2_dropout_reg_l2_001_middle01_97 = load_model('./final-model/model_2_dropout_reg-l2-0.01_middle01-97.h5')

model_2_128_128_64 = load_model('./other-models/model_2_128_128_64/model.h5')
model_dropout__reg_l2_001_middle01_95 = load_model('./other-models/model_dropout__reg-l2-0.01_middle01/model.h5')
model_dropout_input02_92 = load_model('./other-models/model_dropout_input02/model.h5')
model_dropout_middle01_96 = load_model('./other-models/model_2_128_128_64/model.h5')

models = [model_2_128_128_64, model_dropout__reg_l2_001_middle01_95, model_dropout_input02_92]


# %% Layers activations

def visualize_activations(model, path):    
    layer_outputs = [layer.output for layer in model.layers]
    activation_model = Model(inputs=model.input, outputs=layer_outputs)
    
    def display_activation(activations, col_size, row_size, act_index):
        activation = activations[act_index]
        activation_index=0
        fig, ax = plt.subplots(row_size, col_size, figsize=(row_size*2.5,col_size*1.5))
        plt.suptitle("Activations")
        for row in range(0,row_size):
            for col in range(0,col_size):
                ax[row][col].imshow(activation[activation_index, :, :, 0], cmap='gray')
                activation_index+= 1
    
        plt.savefig(f'{path}/activations.jpg')
    
    activations = activation_model.predict(test_generator[3][0])
    display_activation(activations, 9, 5, 0)
    
#%% Kernel Heatmap

def kernel_heatmap(model, layer_name, path):
    kernels = model.get_layer(layer_name).get_weights()[0][:, :, 0, :] 
    
    fig = plt.figure(figsize=(12,12))
    plt.suptitle("{}: Kernels Heatmap".format(layer_name))
    
    n_filters = kernels.shape[2]
    for i in range(n_filters):
        ax = fig.add_subplot(6,6, i+1)
        imgplot = seaborn.heatmap(kernels[:,:,i])
        ax.set_title('Kernel No. {}'.format(i))
        ax.set_aspect('equal', adjustable='datalim')

    fig.tight_layout()
    plt.savefig(f'{path}/kernel_heatmap_{layer_name}.jpg')
    plt.show()

# %% Confusion matrix

def get_confusion_matrix(model):
    '''
    This functions calculates the confusion matrix cm for the model model
    '''
    probabilities = model.predict_generator(generator=test_generator)
    y_true = test_generator.classes
    y_pred = probabilities > 0.5

    cm = confusion_matrix(y_true, y_pred)
    return cm

# Function provided by https://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html#sphx-glr-auto-examples-model-selection-plot-confusion-matrix-py
def plot_confusion_matrix(cm, path):
    """
    This function prints and plots the confusion matrix.
    """
    classes = ['pneumonia', 'normal']
    plt.figure()
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
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
    plt.savefig(f'{path}/confusion_matrix.jpg')
    plt.show()


# %% Visualizations

for model in models:
    path = f'./other-models/{model.name}'
    visualize_activations(model, path)
    kernel_heatmap(model, 'conv_2d_1', path)
    cm = get_confusion_matrix(model)
    plot_confusion_matrix(cm, path)

visualize_activations(
    model_dropout_middle01_96,
    './other-models/model_dropout_middle01')
kernel_heatmap(
    model_dropout_middle01_96,
    'conv_2d_1',
    './other-models/model_dropout_middle01')
plot_confusion_matrix(
    get_confusion_matrix(model_dropout_middle01_96),
    './other-models/model_dropout_middle01')

visualize_activations(
    model_2_dropout_reg_l2_001_middle01_97,
    './final-model')
kernel_heatmap(
    model_2_dropout_reg_l2_001_middle01_97,
    'conv_2d_1',
    './final-model')
plot_confusion_matrix(
    get_confusion_matrix(model_2_dropout_reg_l2_001_middle01_97),
    './final-model')
    
# %%
