from typing import List

import keras
import numpy as np
from keras.callbacks import History
from keras.models import load_model
from keras.optimizers import Optimizer


class Seq2Seq(object):
    def __init__(self, optimizer: Optimizer, loss, layers: List[int] = (35, 35),
                 decay: float = 0., num_input_features: int = 1, num_output_features: int = 1):
        # Encoder part
        encoder_input_layer = keras.layers.Input(shape=(None, num_input_features))
        # Create a list of RNN Cells, these are then concatenated into a single layer
        # with the RNN layer.
        encoder_neurons = []
        for hidden_neurons in layers:
            encoder_neurons.append(keras.layers.GRUCell(hidden_neurons))
        encoder = keras.layers.RNN(encoder_neurons, return_state=True)

        # Discard encoder outputs and only keep the states, we are not interested in the
        # output of the encoder
        encoder_outputs_and_states = encoder(encoder_input_layer)
        encoder_states = encoder_outputs_and_states[1:]

        # The decoder input will be set to zero, only interested in the states of the encoder
        decoder_input_layer = keras.layers.Input(shape=(None, 1))

        decoder_neurons = []
        for hidden_neurons in layers:
            decoder_neurons.append(keras.layers.GRUCell(hidden_neurons))
        decoder = keras.layers.RNN(decoder_neurons, return_sequences=True, return_state=True)

        # Set the initial state of the decoder to be the output state of the encoder.
        # This is the fundamental part of the encoder-decoder.
        decoder_outputs_and_states = decoder(decoder_input_layer, initial_state=encoder_states)

        # Only select the output of the decoder (not the states)
        decoder_outputs = decoder_outputs_and_states[0]

        # Apply a dense layer with linear activation to set output to correct dimension
        # and scale (tanh is default activation for GRU in Keras, our output sine function
        # can be larger then 1)
        decoder_dense = keras.layers.Dense(num_output_features, activation='linear')
        decoder_outputs = decoder_dense(decoder_outputs)

        self.model = keras.models.Model(inputs=[encoder_input_layer, decoder_input_layer],
                                        outputs=decoder_outputs)
        self.model.compile(optimizer=optimizer, loss=loss)

        # Prediction model
        self.encoder_predict_model = keras.models.Model(encoder_input_layer, encoder_states)

        decoder_states_inputs = []

        for hidden_neurons in layers[::-1]:
            decoder_states_inputs.append(keras.layers.Input(shape=(hidden_neurons,)))

        decoder_outputs_and_states = decoder(decoder_input_layer, initial_state=decoder_states_inputs)

        decoder_outputs = decoder_outputs_and_states[0]
        decoder_states = decoder_outputs_and_states[1:]

        decoder_outputs = decoder_dense(decoder_outputs)

        self.decoder_predict_model = keras.models.Model([decoder_input_layer] + decoder_states_inputs,
                                                        [decoder_outputs] + decoder_states)

    def fit(self, encoder_input: np.ndarray, decoder_output: np.ndarray, epochs: int = 100,
            batch_size: int = None, steps_per_epoch: int = None) -> History:
        # The decoder does not have any input, but has to be generated
        decoder_input = np.zeros((encoder_input.shape[0], 1, 1))
        return self.model.fit([encoder_input, decoder_input], decoder_output, epochs=epochs,
                              batch_size=batch_size, steps_per_epoch=steps_per_epoch)

    def predict(self, x_input, num_steps_to_predict):
        """
        Predict time series with encoder-decoder.

        Uses the encoder and decoder models previously trained to predict the next
        num_steps_to_predict values of the time series.

        :param x_input: input time series of shape (batch_size, input_sequence_length,
        input_dimension).
        :param num_steps_to_predict: The number of steps in the future to predict
        :return: y_predicted: output time series for shape
        (batch_size, target_sequence_length, ouput_dimension)
        """
        y_predicted = []

        states = self.encoder_predict_model.predict(x_input)
        if not isinstance(states, list):
            states = [states]

        decoder_input = np.zeros((x_input.shape[0], 1, 1))

        for _ in range(num_steps_to_predict):
            outputs_and_states = self.decoder_predict_model.predict([decoder_input] + states,
                                                                    batch_size=x_input.shape[0])
            output = outputs_and_states[0]
            states = outputs_and_states[1:]

            # add predicted value
            y_predicted.append(output)

        return np.concatenate(y_predicted, axis=1)

    def save_model(self, name):
        self.model.save('ckpt/model_{0}.h5'.format(name))

    def load_model(self, name):
        self.model = load_model('ckpt/model_{0}.h5'.format(name))
