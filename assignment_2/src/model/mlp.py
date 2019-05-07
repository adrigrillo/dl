from typing import Tuple

import numpy as np
from keras import Input, Model
from keras.callbacks import History
from keras.engine.saving import load_model
from keras.layers import Dense, regularizers, Dropout, BatchNormalization
from keras.optimizers import Optimizer


class MLP:
    def __init__(self, input_size: Tuple, optimizer: Optimizer, loss, hidden_layers: Tuple = (3, 3, 1),
                 activation: str = 'relu', output_activation: str = 'relu',
                 dropout: float = 0., batch_normalization: bool = False,
                 weight_decay_l1: float = 0., weight_decay_l2: float = 0.):
        # define model
        self.hidden_layers = hidden_layers

        # create the model
        inputs = x_data = Input(shape=input_size)
        # rest of the hidden layers if any
        for neurons in hidden_layers[:-1]:
            x_data = Dense(neurons, activation=activation,
                      kernel_regularizer=regularizers.l1_l2(l1=weight_decay_l1, l2=weight_decay_l2),
                      bias_regularizer=regularizers.l1_l2(l1=weight_decay_l1, l2=weight_decay_l2))(x_data)
            if dropout > 0.:
                x_data = Dropout(dropout)(x_data)
            if batch_normalization:
                x_data = BatchNormalization()(x_data)
        predictions = Dense(hidden_layers[-1], activation=output_activation)(x_data)
        self.model = Model(inputs=inputs, outputs=predictions)
        self.model.compile(optimizer=optimizer, loss=loss)

    def fit(self, inputs: np.ndarray, outputs: np.ndarray, batch_size: int = 1, epochs: int = 100) -> History:
        return self.model.fit(inputs, outputs, batch_size=batch_size, epochs=epochs)

    def predict(self, data, number_of_steps):
        predictions = np.empty(shape=(number_of_steps,))
        data_shape = data.shape

        for i in range(predictions.shape[0]):
            predicted_value = self.model.predict(data)
            predictions[i] = predicted_value.item()
            # remove first element and add the prediction
            data = np.reshape(np.append(data[0][1:], predicted_value.item()), newshape=data_shape)
        return predictions

    def save_model(self, name):
        self.model.save('ckpt/' + name)

    def load_model(self, name):
        self.model = load_model(name)
