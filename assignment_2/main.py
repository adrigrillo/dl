# %%
import keras
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat

# %% Load and plot the y
from utils import process_data, generate_train_and_validation_sets

time_series: np.ndarray = loadmat('data/Xtrain.mat')['Xtrain'].squeeze()
data_points = time_series.shape[0]

# %%
plt.figure(1, figsize=[15, 5])
plt.plot(time_series)
plt.xlabel('x')
plt.xlabel('y')
plt.title('Time series data')
plt.show()

# %%
x, y = process_data(time_series, 3)
x_train_folds, y_train_folds, x_test_folds, y_test_folds = generate_train_and_validation_sets(x, y, folds=10)

# %% encoder-decoder method
keras.backend.clear_session()
# Number of hidden neurons in each layer of the encoder and decoder
layers = [35, 35]
optimiser = keras.optimizers.Adam(lr=0.01, decay=0)
loss = "mse"
# The dimensionality of the input and output
num_input_features = 1
num_output_features = 1

encoder_input_layer = keras.layers.Input(shape=(None, num_input_features))

# Create a list of RNN Cells, these are then concatenated into a single layer
# with the RNN layer.
encoder_neurons = []
for hidden_neurons in layers:
    encoder_neurons.append(keras.layers.GRUCell(hidden_neurons))

encoder = keras.layers.RNN(encoder_neurons, return_state=True)

encoder_outputs_and_states = encoder(encoder_input_layer)

# Discard encoder outputs and only keep the states.
# The outputs are of no interest to us, the encoder's
# job is to create a state describing the input sequence.
encoder_states = encoder_outputs_and_states[1:]

# The decoder input will be set to zero (see random_sine function of the utils module).
# Do not worry about the input size being 1, I will explain that in the next cell.
decoder_input_layer = keras.layers.Input(shape=(None, 1))

decoder_neurons = []
for hidden_neurons in layers:
    decoder_neurons.append(keras.layers.GRUCell(hidden_neurons))

decoder = keras.layers.RNN(decoder_neurons, return_sequences=True, return_state=True)

# Set the initial state of the decoder to be the ouput state of the encoder.
# This is the fundamental part of the encoder-decoder.
decoder_outputs_and_states = decoder(decoder_input_layer, initial_state=encoder_states)

# Only select the output of the decoder (not the states)
decoder_outputs = decoder_outputs_and_states[0]

# Apply a dense layer with linear activation to set output to correct dimension
# and scale (tanh is default activation for GRU in Keras, our output sine function can be larger then 1)
decoder_dense = keras.layers.Dense(num_output_features, activation='linear')

decoder_outputs = decoder_dense(decoder_outputs)

model = keras.models.Model(inputs=[encoder_input_layer, decoder_input_layer], outputs=decoder_outputs)
model.compile(optimizer=optimiser, loss=loss)

# %%
train_input = list()
for index in range(x.shape[0]):
    input_decoder = np.zeros(x[index].shape)
    train_input.append(np.array([x[index], input_decoder]))

# %%
model.fit(train_input[0], y[0], steps_per_epoch=100, epochs=100)

