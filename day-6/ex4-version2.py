import seaborn
from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten
from tensorflow.keras import models
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.regularizers import l2
import matplotlib.pyplot as plt

# %% Load the data

(train_data, train_labels), (test_data, test_labels) = mnist.load_data()

seaborn.heatmap(train_data[0, :, :])

train_data = train_data.reshape((len(train_data), 28, 28, 1)) / 255
test_data = test_data.reshape((len(test_data), 28, 28, 1)) / 255

train_labels = to_categorical(train_labels, num_classes=10)
test_labels = to_categorical(test_labels, num_classes=10)

# %% Building and compiling the model

model = models.Sequential()

model.add(Conv2D(32, (5, 5),
          activation='relu',
          name='first_layer',
          input_shape=(28, 28, 1),
          data_format='channels_last'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (5, 5), activation='relu', data_format='channels_last'))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dense(10, activation='softmax'))

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# %% Plotting before the training

kernels = model.get_layer(name='first_layer').get_weights()[0][:, :, 0, :]

# %% Fitting the model

history = model.fit(train_data, train_labels, epochs=2,
                    validation_data=(test_data, test_labels))

# %% Plotting after the training

kernels_after = model.get_layer(
    name='first_layer').get_weights()[0][:, :, 0, :]

# %% Activations

layer_outputs = [layer.output for layer in model.layers]
activation_model = models.Model(inputs=model.input, outputs=layer_outputs)
activations = activation_model.predict(train_data[10].reshape(1, 28, 28, 1))


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


display_activation(activations, 6, 5, 0)

# %%
