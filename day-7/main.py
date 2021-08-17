# %% Imports
import tensorflow
import numpy as np
from tensorflow.keras import datasets
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input, Concatenate, Embedding, LSTM, Dropout, Flatten, SimpleRNN
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.regularizers import l2
import matplotlib.pyplot as plt

# %% Loading the data

NUM_WORDS = 10000
(X_train, y_train), (X_test, y_test) = datasets.imdb.load_data(num_words=NUM_WORDS)

MAXLEN = 500
WORD_VEC_SIZE = 32

# Making the sequences all MAXLEN words long
X_train = pad_sequences(X_train, maxlen=MAXLEN)
X_test = pad_sequences(X_test, maxlen=MAXLEN)

# %% Defining the model

inputs = Input(shape=(MAXLEN,), name='IMDB_input')

x = inputs
x = Embedding(NUM_WORDS, WORD_VEC_SIZE, name='embedding')(x)
x = Dropout(0.1, name='dropout')(x)
x = LSTM(64, activation='tanh', return_sequences=True, name='recurrent_1')(x)
x = Flatten(name='flatten')(x)
x = Dense(32, activation='relu', kernel_regularizer=l2(0.01), name='1st_dense')(x)

left = Dense(32, activation='relu', kernel_regularizer=l2(0.01), name='1st_left')(x)

right = Dense(32, activation='relu', kernel_regularizer=l2(0.01), name='1st_right')(x)

x = Concatenate(name='concat')([left, right])
output = Dense(1, activation='sigmoid', name='sentiment_output')(x)

model = Model(inputs=inputs, outputs=output, name='rnn_model')

model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy'])

print(model.summary())

# %% Try and fit the model

history = model.fit(X_train, y_train, epochs=10, batch_size=128, validation_data=(X_test, y_test))

# %% Accuracy graph

accuracies = history.history['accuracy']
val_accuracies = history.history['val_accuracy']
epochs = range(1, len(accuracies) + 1)

plt.figure()
plt.plot(epochs, accuracies, label='Training')
plt.plot(epochs, val_accuracies, label='Validation')
plt.title('Accuracy')
plt.legend()
plt.show()

# %% Loss graph

losses = history.history['loss']
val_losses = history.history['val_loss']
epochs = range(1, len(losses) + 1)

plt.figure()
plt.plot(epochs, losses, label='Training')
plt.plot(epochs, val_losses, label='Validation')
plt.title('Loss')
plt.legend()
plt.show()

# %%
