# %%
import keras
import numpy as np
from keras.optimizers import Adam
from keras.utils import plot_model

from src.model.seq_to_seq import Seq2Seq
from src.utils.data_handling import process_data, normalize_data
from src.utils.plotting import plot_series
from src.utils.training import root_mean_square_error

# %% Load

time_series: np.ndarray = np.loadtxt('data/Xtrain.csv', delimiter=',')
time_series = np.reshape(time_series, newshape=(time_series.shape[0], 1))
data_points = time_series.shape[0]

# %%
time_series = normalize_data(time_series)
x, y = process_data(time_series, x_window_size=200, y_window_size=200, dimensions=3)
end_train = 1000
end_test = 1200
x_test = x[end_train:end_test]
y_test = y[end_train:end_test]
x = x[:end_train]
y = y[:end_train]

# %%
keras.backend.clear_session()
optimizer = Adam(0.01)
loss = root_mean_square_error
model = Seq2Seq(optimizer, loss)
plot_model(model.model, to_file='images/model.png', show_shapes=True)
# %%
model.fit(x, y, epochs=200)
#%%
x_test_1 = np.reshape(x_test[0], newshape=(1, 200, 1))
x_decoder_test = np.zeros(shape=x_test_1.shape)
y_test_predicted = model.model.predict([x_test_1, x_decoder_test])
#%%
plot_series(y_test_predicted.squeeze())
# %%
decoder_input = np.zeros((x_test.shape[0], 1, 1))
model.model.evaluate([x_test, decoder_input], y_test)
