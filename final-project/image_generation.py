from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator

img_dims = 500   #we can still change this
batch_size = 32

model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape = (img_dims, img_dims, 3), activation = 'relu')) # this helps the tensor of output
model.add(MaxPooling2D(pool_size = (2, 2))) # i used pooling since we need max value pixel for the ROI
model.add(Flatten()) #then flatten
model.add(Dense(128, activation = 'relu'))
model.add(Dense(1, activation = 'sigmoid'))

model.compile(optimizer = 'adam',
                   loss = 'binary_crossentropy',
                   metrics = ['accuracy'])

# we start with the image data generator
training_datagenerator = ImageDataGenerator(rescale=1/255,
                                   rotation_range=10,
                                   zoom_range=0.2,
                                   shear_range=0.5,
                                   width_shift_range=0.1,
                                   height_shift_range=0.1)
# i used the rescale parameter for grayscale normalization
# so it basically transforms the pixels from [0…255] to [0…1]

testing_datagenerator = ImageDataGenerator(rescale=1/255)

# as stated, using the .flow_from_directory makes it work with the directory in task 1
train_generator = training_datagenerator.flow_from_directory(
    '/Users/mac/PycharmProjects/pythonProject/data/training',
    target_size = (256,256), # refers to the size width and height
    batch_size = 32,
    class_mode = 'binary')

validation_generator = testing_datagenerator.flow_from_directory(
    '/Users/mac/PycharmProjects/pythonProject/data/validation',
    target_size = (256, 256),
    batch_size = 32,
    class_mode = 'binary')

# training the model
history = model.fit_generator(train_generator,
                    steps_per_epoch = 10,
                    epochs = 10,
                    validation_data = validation_generator)

#after training the model, we introduce the unseen data from the testing
#by loading the testing dataset

test_generator = testing_datagenerator.flow_from_directory(
    '/Users/mac/PycharmProjects/pythonProject/data/testing',
    target_size = (256, 256),
    batch_size = 32,
    class_mode = 'binary')

eval_result = model.evaluate_generator(test_generator, 624)

# we can now see the accuracy of the model by running this
print('loss rate at evaluation data :', eval_result[0])
print('accuracy rate at evaluation data :', eval_result[1])