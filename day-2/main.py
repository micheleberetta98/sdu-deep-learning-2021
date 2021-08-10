import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers

NUM_WORDS = 10000
(train_data, train_labels), (test_data,
                             test_labels) = keras.datasets.imdb.load_data(num_words=NUM_WORDS)


def multi_hot_sequences(sequences, dimension):
    # Create an all-zero matrix of shape (len(sequences), dimension)
    results = np.zeros((len(sequences), dimension))
    for i, word_indices in enumerate(sequences):
        # set specific indices of results[i] to 1s
        results[i, word_indices] = 1.0
    return results


# train_data has shape (25_000, NUM_WORDS)
train_data = multi_hot_sequences(train_data, dimension=NUM_WORDS)
test_data = multi_hot_sequences(test_data, dimension=NUM_WORDS)


def build_model(first, second):
    model = keras.Sequential()
    model.add(keras.layers.Dense(
        first, activation='relu', input_shape=(NUM_WORDS,)))
    model.add(keras.layers.Dense(second, activation='relu'))
    model.add(keras.layers.Dense(1, activation='sigmoid'))
    return model


def compile(model):
    model.compile(optimizer='rmsprop',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])


def fit_model(model, epochs=5):
    history = model.fit(train_data, train_labels,
                        epochs=epochs,
                        validation_data=(test_data, test_labels))
    return history


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


model = build_model(16, 16)
compile(model)
history = fit_model(model, 20)
model.save('model-1-relu-relu-sigm.h5')

fig, (subfig1, subfig2) = plt.subplots(2, 1)
fig.suptitle('Model with (16, 16, 1) units')
show_plot_accuracy(subfig1, history)
show_plot_loss(subfig2, history)
fig.show()

# 1. What accuracy do you get?
#    With 1 epoch, around 87% per the training data, and about 85% for the validation accuracy
#    With more epochs, the validation accuracy tends to go higher, as the validation loss

#    With more epochs, the accuracy seems to be way higher, closer to 98%, wich could be an indication
#    of overfitting, the validation loss value is really high

print('== ACCURACY ==')
print(history.history['accuracy'])

print('== VALIDATION ACCURACY ==')
print(history.history['val_accuracy'])

# 2. Do you think it's a good result?
#    Seems good, but the validation loss keeps going up - may be a sign of overfitting

# 3. Try and change the units
#    Lowering the first layer doesn't seem to affect the network a lot
#    Neither making the units go up (in both layers)
#    Maybe because of overfitting, changing the model (as far as changing the units go) doesn't seem
#    to affect the model

#  Testing values 5,10,15,20,25

for i in range(5):
    for j in range(5):
        k1 = (i + 1) * 5
        k2 = (j + 1) * 5

        print(f'==> Checking model with ({k1}, {k2}, 1)')
        m = build_model(k1, k2)
        compile(m)
        h = fit_model(m)

        print('ACCURACY (testing, validation)')
        print(h.history['accuracy'])
        print(h.history['val_accuracy'])
        print('LOSS (testing, validation)')
        print(h.history['loss'])
        print(h.history['val_loss'])
