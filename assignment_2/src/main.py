# %%
import keras
import numpy as np
from keras.optimizers import Adam
from keras.utils import plot_model

from src.model.mlp import MLP
from src.utils.data_handling import process_data, normalize_data
from src.utils.plotting import plot_comparison
from src.utils.training import root_mean_square_error

# %% Load

# time_series: np.ndarray = loadmat('data/Xtrain.mat')['Xtrain']
time_series: np.ndarray = np.loadtxt('data/Xtrain.csv', delimiter=',')
data_points = time_series.shape[0]

# %%
time_series = normalize_data(time_series)
windows = 30
x, y = process_data(time_series, x_window_size=windows, y_window_size=1, dimensions=2)

# %% split train-test
end_train = 1000
end_test = 1200
x_test = x[end_train:end_test]
y_test = y[end_train:end_test]
x = x[:end_train]
y = y[:end_train]

# %% encoder-decoder method
keras.backend.clear_session()
optimizer = Adam(0.01)
loss = root_mean_square_error
model = MLP(input_size=(windows,), optimizer=optimizer, loss=loss,
            hidden_layers=(10, 5, 1), activation='tanh', output_activation='tanh',
            dropout=0.1)

# %%
history = model.fit(x, y, batch_size=1, epochs=10)
# %%
val_error = model.model.evaluate(x_test, y_test)
print('The validation error using x_test and y_test is {0}'.format(val_error))
# %%
test_1 = np.reshape(x_test[0], newshape=(1, windows))
predictions = model.predict(test_1, 200)

# %% plot
plot_comparison(predictions, y_test)
plot_model(model.model, to_file='images/model.png', show_shapes=True)


# %% SEQ-2-SEQ
def plot_prediction(x, y_true, y_pred):
    """Plots the predictions.

    Arguments
    ---------
    x: Input sequence of shape (input_sequence_length,
        dimension_of_signal)
    y_true: True output sequence of shape (input_sequence_length,
        dimension_of_signal)
    y_pred: Predicted output sequence (input_sequence_length,
        dimension_of_signal)
    """

    plt.figure(figsize=(12, 3))

    output_dim = x.shape[-1]
    for j in range(output_dim):
        past = x[:, j]
        true = y_true[:, j]
        pred = y_pred[:, j]

        label1 = "Seen (past) values" if j == 0 else "_nolegend_"
        label2 = "True future values" if j == 0 else "_nolegend_"
        label3 = "Predictions" if j == 0 else "_nolegend_"

        plt.plot(range(len(past)), past, "o--b",
                 label=label1)
        plt.plot(range(len(past),
                       len(true) + len(past)), true, "x--b", label=label2)
        plt.plot(range(len(past), len(pred) + len(past)), pred, "o--y",
                 label=label3)
    plt.legend(loc='best')
    plt.title("Predictions v.s. true values")
    plt.show()
