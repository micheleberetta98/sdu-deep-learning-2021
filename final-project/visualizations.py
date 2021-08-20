# -*- coding: utf-8 -*-
"""
Visualizing
Created on Thu Aug 19 21:35:19 2021

@author: Admin
"""
 
 # %%

from image_generation import validation_generator, train_generator, test_generator
from tensorflow.keras.models import Model, load_model
import matplotlib.pyplot as plt
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import Model
from numpy import expand_dims

import seaborn


model_2_dropout_reg_l2_001_middle01_97 = load_model('./final-model/model_2_dropout_reg-l2-0.01_middle01-97.h5')

model_2_128_128_64 = load_model('./other-models/model_2_128_128_64/model.h5')
model_dropout__reg_l2_001_middle01_95 = load_model('./other-models/model_dropout__reg-l2-0.01_middle01/model.h5')
model_dropout_input02_92 = load_model('./other-models/model_dropout_input02/model.h5')
model_dropout_middle01_96 = load_model('./other-models/model_2_128_128_64/model.h5')

models = [model_2_128_128_64, model_dropout__reg_l2_001_middle01_95, model_dropout_input02_92]


# %%

# visualize feature maps output from each block in the vgg model

# redefine model to output right after the first hidden layer
ixs = [2, 5, 9, 13, 17]
outputs = [model.layers[i].output for i in ixs]
model = Model(inputs=model.inputs, outputs=outputs)
# load the image with the required shape
img = load_img('bird.jpg', target_size=(224, 224))
# convert the image to an array
img = img_to_array(img)
# expand dimensions so that it represents a single 'sample'
img = expand_dims(img, axis=0)
# prepare the image (e.g. scale pixel values for the vgg)
img = preprocess_input(img)
# get feature map for first hidden layer
feature_maps = model.predict(img)
# plot the output from each block
square = 8
for fmap in feature_maps:
	# plot all 64 maps in an 8x8 squares
	ix = 1
	for _ in range(square):
		for _ in range(square):
			# specify subplot and turn of axis
			ax = pyplot.subplot(square, square, ix)
			ax.set_xticks([])
			ax.set_yticks([])
			# plot filter channel in grayscale
			pyplot.imshow(fmap[0, :, :, ix-1], cmap='gray')
			ix += 1
	# show the figure
	pyplot.show()







# %% Visualize Activations

def visualize_activations(model):    
    layer_outputs= [layer.output for layer in model.layers]
    activation_model= Model(inputs=model.input, outputs=layer_outputs)
    
    def display_activation(activations, col_size, row_size, act_index):
        activation = activations[act_index]
        activation_index=0
        fig, ax = plt.subplots(row_size, col_size, figsize=(row_size*2.5,col_size*1.5))
        plt.suptitle("{}: Activations".format(model.name))
        for row in range(0,row_size):
            for col in range(0,col_size):
                ax[row][col].imshow(activation[activation_index, :, :, 0], cmap='gray')
                activation_index+= 1
    
        plt.savefig('{}_activations'.format(model.name))
    
    activations = activation_model.predict(test_generator[3])
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
    
# %%
for model in models:
    # visualize_activations(model)
    kernel_heatmap(model, 'conv_2d_1', f'./other-models/{model.name}')

kernel_heatmap(
    model_dropout_middle01_96,
    'conv_2d_1',
    './other-models/model_dropout_middle01'
)

kernel_heatmap(
    model_2_dropout_reg_l2_001_middle01_97,
    'conv_2d_1',
    './final-model'
)
    
# %%
