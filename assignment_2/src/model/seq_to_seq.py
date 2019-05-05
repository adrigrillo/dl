from typing import List

import keras
import numpy as np
from keras.callbacks import History
from keras.models import load_model

FOLDER = 'ckpt/'


class Seq2Seq(object):
    def __init__(self, layers: List[int] = (35, 35), lr: float = 0.01, decay: float = 0,
                 loss: str = 'mse', num_input_features: int = 1, num_output_features: int = 1):
        self.optimiser = keras.optimizers.Adam(lr=lr, decay=decay)
        self.loss = loss

        # Encoder part
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

        self.model = keras.models.Model(inputs=[encoder_input_layer, decoder_input_layer], outputs=decoder_outputs)
        self.model.compile(optimizer=self.optimiser, loss=loss)

        # Prediction model
        self.encoder_predict_model = keras.models.Model(encoder_input_layer, encoder_states)

    def fit(self, encoder_input: np.ndarray, decoder_output: np.ndarray, epochs: int = 100) -> History:
        # The decoder does not have any input, but has to be generated
        decoder_input = np.zeros((encoder_input.shape[0], 1, 1))
        return self.model.fit([encoder_input, decoder_input], decoder_output, epochs=epochs)

    def predict(self, x, num_steps_to_predict):
        """
        Predict time series with encoder-decoder.

        Uses the encoder and decoder models previously trained to predict the next
        num_steps_to_predict values of the time series.


        :param x: input time series of shape (batch_size, input_sequence_length, input_dimension).
        :param num_steps_to_predict: The number of steps in the future to predict
        :return y_predicted: output time series for shape (batch_size, target_sequence_length, ouput_dimension)
        """
        # TODO: implement this well

    def save_model(self, name):
        self.model.save(FOLDER + 'model_{0}.h5'.format(name))

    def load_model(self, name):
        self.model = load_model(FOLDER + 'model_{0}.h5'.format(name))
