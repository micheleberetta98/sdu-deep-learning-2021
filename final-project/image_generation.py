'''
    This file defines all the images generators to use in the model creation
    @author Mark John Paul Pangan
'''

from tensorflow.keras.preprocessing.image import ImageDataGenerator

from constants import IMG_SIZE, BATCH_SIZE, NUM_OF_IMAGES_TRAIN, NUM_OF_IMAGES_VAL

# model = Sequential()
# model.add(Conv2D(32, (3, 3), input_shape=(IMG_SIZE, IMG_SIZE, 1),
#           activation='relu'))  # this helps the tensor of output
# # i used pooling since we need max value pixel for the ROI
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Flatten())  # then flatten
# model.add(Dense(128, activation='relu'))
# model.add(Dense(1, activation='sigmoid'))

# model.compile(optimizer='adam',
#               loss='binary_crossentropy',
#               metrics=['accuracy'])

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
