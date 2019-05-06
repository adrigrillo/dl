from collections import deque
from typing import Tuple

import numpy as np
from keras import Input, Model
from keras.callbacks import History
from keras.engine.saving import load_model
from keras.layers import Dense, regularizers, Dropout, BatchNormalization
from keras.optimizers import Optimizer


class MLP:
    def __init__(self, input_size: Tuple, optimizer: Optimizer, loss, hidden_layers: Tuple[int] = (3, 3),
                 activation: str = 'relu', output_activation: str = 'linear',
                 dropout: float = 0., batch_normalization: bool = False,
                 weight_decay_l1: float = 0., weight_decay_l2: float = 0.):
        # define model
        self.hidden_layers = hidden_layers

        # create the model
        inputs = x = Input(shape=input_size)
        # rest of the hidden layers if any
        for neurons in hidden_layers:
            x = Dense(neurons, activation=activation,
                      kernel_regularizer=regularizers.l1_l2(l1=weight_decay_l1, l2=weight_decay_l2),
                      bias_regularizer=regularizers.l1_l2(l1=weight_decay_l1, l2=weight_decay_l2))(x)
            if dropout > 0.:
                x = Dropout(dropout)(x)
            if batch_normalization:
                x = BatchNormalization()(x)
        predictions = Dense(1, activation=output_activation)(x)
        self.model = Model(inputs=inputs, outputs=predictions)
        self.model.compile(optimizer=optimizer, loss=loss)

    def fit(self, inputs: np.ndarray, outputs: np.ndarray, epochs: int = 100) -> History:
        return self.model.fit(inputs, outputs, epochs=epochs)

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
        self.model.save('saved-models/' + name)

    def load_model(self, name):
        self.model = load_model(name)
