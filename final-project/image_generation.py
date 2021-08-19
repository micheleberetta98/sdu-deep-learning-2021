'''
    This file defines all the images generators to use in the model creation
    @author Mark John Paul Pangan
'''

from tensorflow.keras.preprocessing.image import ImageDataGenerator

from constants import IMG_SIZE, BATCH_SIZE

# we start with the image data generator
training_datagenerator = ImageDataGenerator(rescale=1/255,
                                            rotation_range=10,
                                            zoom_range=0.2,
                                            shear_range=0.5,
                                            width_shift_range=0.1,
                                            height_shift_range=0.1)
# i used the rescale parameter for normalization
# so it basically transforms the pixels from [0…255] to [0…1]

testing_datagenerator = ImageDataGenerator(rescale=1/255)

# as stated, using the .flow_from_directory makes it work with the directory in task 1
train_generator = training_datagenerator.flow_from_directory(
    './training',
    color_mode='grayscale',
    target_size=(IMG_SIZE, IMG_SIZE),  # refers to the size width and height
    batch_size=BATCH_SIZE,
    class_mode='binary')

validation_generator = testing_datagenerator.flow_from_directory(
    './validation',
    color_mode='grayscale',
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='binary')

test_generator = testing_datagenerator.flow_from_directory(
    './testing',
    color_mode='grayscale',
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='binary')
