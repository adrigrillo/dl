# %%
import keras
import numpy as np
from keras.optimizers import Adam
from keras.utils import plot_model

from src.model.mlp import MLP
from src.utils.data_handling import process_data, normalize_data
from src.utils.training import root_mean_square_error

# %% Load

# time_series: np.ndarray = loadmat('data/Xtrain.mat')['Xtrain']
time_series: np.ndarray = np.loadtxt('data/Xtrain.csv', delimiter=',')
data_points = time_series.shape[0]

# %%
time_series = normalize_data(time_series)
x, y = process_data(time_series, window_size=20)

# %% split train-test
x_test = x[1000:1200]
y_test = y[1000:1200]
x = x[:1000]
y = y[:1000]

# %% encoder-decoder method
keras.backend.clear_session()
optimizer = Adam(0.01)
loss = root_mean_square_error
model = MLP(hidden_layers=(3,4), optimizer=optimizer, loss=loss, activation='sigmoid')

# %%
history = model.fit(x, y)
# %%
test_1 = np.reshape(x_test[0], newshape=(1, 20, 1))
y_pred = model.model.predict(test_1)

# %%
evaluation = model.model.evaluate(x_test, y_test)
# %% plot
# plot_comparison(prediction, y[800:].squeeze())
plot_model(model.model, to_file='images/model.png', show_shapes=True)
