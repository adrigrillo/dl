from typing import Tuple

from keras import Input, Model
from keras.callbacks import History
from keras.engine.saving import load_model
from keras.layers import Conv1D, Flatten, Dense, np
from keras.optimizers import Optimizer


class CNN:
    def __init__(self, input_size: Tuple, optimizer: Optimizer, loss,
                 activation: str = 'relu', output_activation: str = None):
        inputs = Input(shape=input_size)
        x_data = Conv1D(filters=64, kernel_size=16, activation=activation)(inputs)
        x_data = Conv1D(filters=32, kernel_size=8, activation=activation)(x_data)
        x_data = Conv1D(filters=16, kernel_size=4, activation=activation)(x_data)
        x_data = Flatten()(x_data)
        outputs = Dense(1, activation=output_activation)(x_data)
        self.model = Model(inputs=inputs, outputs=outputs)
        self.model.compile(optimizer=optimizer, loss=loss)

    def fit(self, inputs: np.ndarray, outputs: np.ndarray,
            epochs: int = 100, batch_size: int = None) -> History:
        return self.model.fit(inputs, outputs, epochs=epochs, batch_size=batch_size)

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
