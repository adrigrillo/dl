from typing import Tuple

import numpy as np
from keras import Input, Model
from keras.callbacks import History
from keras.engine.saving import load_model
from keras.layers import Dense, regularizers, Dropout, BatchNormalization
from keras.optimizers import Optimizer


class MLP:
    def __init__(self, optimizer: Optimizer, loss,
                 hidden_layers: Tuple[int] = (3, 3), activation: str = 'relu',
                 num_input_features: int = 1, num_output_features: int = 1,
                 dropout: float = 0., batch_normalization: bool = False,
                 weight_decay_l1: float = 0., weight_decay_l2: float = 0.):
        # define model
        self.num_input_features = num_input_features
        self.num_output_features = num_output_features
        self.hidden_layers = hidden_layers

        # create the model
        inputs = Input(shape=(None, self.num_input_features))
        # first layer
        x = Dense(hidden_layers[0], activation=activation,
                  kernel_regularizer=regularizers.l1_l2(l1=weight_decay_l1, l2=weight_decay_l2),
                  bias_regularizer=regularizers.l1_l2(l1=weight_decay_l1, l2=weight_decay_l2))(inputs)
        if dropout > 0.:
            x = Dropout(dropout)(x)
        if batch_normalization:
            x = BatchNormalization()(x)
        # rest of the hidden layers if any
        for i in range(1, len(hidden_layers)):
            x = Dense(hidden_layers[i], activation=activation,
                      kernel_regularizer=regularizers.l1_l2(l1=weight_decay_l1, l2=weight_decay_l2),
                      bias_regularizer=regularizers.l1_l2(l1=weight_decay_l1, l2=weight_decay_l2))(x)
            if dropout > 0.:
                x = Dropout(dropout)(x)
            if batch_normalization:
                x = BatchNormalization()(x)
        predictions = Dense(1, activation='linear')(x)
        self.model = Model(inputs=inputs, outputs=predictions)
        self.model.compile(optimizer=optimizer, loss=loss)

    def fit(self, inputs: np.ndarray, outputs: np.ndarray, epochs: int = 100) -> History:
        return self.model.fit(inputs, outputs, epochs=epochs)

    def predict(self, input):
        # demonstrate prediction
        x_input = input.reshape((1, self.input_size))
        prediction = self.model.predict(x_input, verbose=0)
        return prediction

    def save_model(self, name):
        self.model.save('saved-models/' + name)

    def load_model(self, name):
        self.model = load_model(name)
