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
x, y = process_data(time_series, window_size=windows, dimensions=2)

# %% split train-test
x_test = x[800:1000]
y_test = y[800:1000]
x = x[:800]
y = y[:800]

# %% encoder-decoder method
keras.backend.clear_session()
optimizer = Adam(0.01)
loss = root_mean_square_error
model = MLP(input_size=(windows,), optimizer=optimizer, loss=loss)

# %%
history = model.fit(x, y, batch_size=1, epochs=10)
# %%
val_error = model.model.evaluate(x_test, y_test)
print('The validation error using x_test and y_test is {0}'.format(val_error))
#%%
test_1 = np.reshape(x_test[0], newshape=(1, windows))
predictions = model.predict(test_1, 200)

# %% plot
plot_comparison(predictions, y_test)
plot_model(model.model, to_file='images/model.png', show_shapes=True)
