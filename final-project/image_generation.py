
# %% Imports

import tensorflow as tf
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout, Input
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# %%
# we start with the image data generator
training_datagenerator = ImageDataGenerator(rescale=1/255,
                                            rotation_range=10,
                                            zoom_range=0.2,
                                            shear_range=0.5,
                                            width_shift_range=0.1,
                                            height_shift_range=0.1)
# i used the rescale parameter for normalization
# so it basically transforms the pixels from [0..255] to [0..1]

testing_datagenerator = ImageDataGenerator(rescale=1/255)

IMG_SIZE = 256
BATCH_SIZE = 45
EPOCHS = 10

inputs = Input(shape=(IMG_SIZE, IMG_SIZE, 1), name='input_layer')
x = inputs
x = Conv2D(32, (5, 5), activation='relu', name='conv_2d_1')(x)
x = Conv2D(32, (5, 5), activation='relu', name='conv_2d_3')(x)
x = MaxPooling2D((3, 3), name='max_2d_2')(x)

x = Flatten(name='flatten')(x)

x = Dense(64, activation='relu', name='dense_1')(x)
x = Dense(64, activation='relu', name='dense_2')(x)
x = Dense(64, activation='relu', name='dense_3')(x)

# Since we are dealing with a binary problem (pneumonia or normal),
# the output is given by this 1 neuron with a sigmoid activation function
output = Dense(1, activation='sigmoid', name='dense_output')(x)

model = Model(inputs=inputs, outputs=output, name='test')

model.compile(optimizer='rmsprop',
              # Binary crossentropy as a loss function is ideal for a binary problem
              loss='binary_crossentropy',
              # We are interested in the network accuracy (and also the loss)
              metrics=['accuracy'])

# as stated, using the .flow_from_directory makes it work with the directory in task 1
train_generator = training_datagenerator.flow_from_directory(
    './training',
    color_mode='grayscale',  # images are all grayscale, so we take out the 3 channels
    target_size=(IMG_SIZE, IMG_SIZE),  # refers to the size width and height
    batch_size=BATCH_SIZE,
    class_mode='binary')

validation_generator = testing_datagenerator.flow_from_directory(
    './validation',
    color_mode='grayscale',
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='binary')

model.fit(train_generator,
          steps_per_epoch=(2 * 900 / BATCH_SIZE),
          epochs=10,
          validation_data=validation_generator,
          callbacks=[])

# after training the model, we introduce the unseen data from the testing
# by loading the testing dataset

test_generator = testing_datagenerator.flow_from_directory(
    './testing',
    color_mode='grayscale',
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='binary')

# %%
