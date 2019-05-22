# %%
import keras
import numpy as np
from keras.optimizers import Adam
from scipy.io import loadmat
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# %% Load
from timeseries_forecasting.src.model.lstm import RNN
from timeseries_forecasting.src.utils.data_handling import process_data
from timeseries_forecasting.src.utils.plotting import plot_comparison
from timeseries_forecasting.src.utils.training import calculate_rmse

time_series: np.ndarray = loadmat('data/Xtrain.mat')['Xtrain']
test_series: np.ndarray = loadmat('data/Xtest.mat')['Xtest']
min_max_scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
time_series = min_max_scaler.fit_transform(time_series)
# %%
windows = 50
x, y = process_data(time_series, x_window_size=windows, y_window_size=1, dimensions=3)
x = np.reshape(x, newshape=(x.shape[0], 1, windows))
y = y.squeeze()

# end_train = int(x.shape[0] * 0.8)
# x_test = x[end_train:]
# y_test = y[end_train:]
# x = x[:end_train]
# y = y[:end_train]
# %%
keras.backend.clear_session()
optimizer = Adam()
loss = 'mse'
model = RNN(input_size=(1, windows), optimizer=optimizer, loss=loss)

history = model.fit(x, y, epochs=150, batch_size=16)
val_error = model.model.evaluate(x, y)
print('The validation error using x_test and y_test is {0}'.format(val_error))
# %%
data = np.reshape(time_series[950:], newshape=(1, 1, 50))
# %%
test = np.reshape(data, newshape=(1, 1, windows))
predictions = model.predict(test, 200)

predictions = np.reshape(predictions, newshape=test_series.shape)
predictions = min_max_scaler.inverse_transform(predictions)

# %%
mse_prediction = mean_squared_error(predictions, test_series)
mae_prediction = mean_absolute_error(predictions, test_series)
rmse_prediction = calculate_rmse(predictions, test_series)
r2 = r2_score(predictions, test_series)
print('The error predicting 200 steps is mse: {0} mae: {1} rmse:{2} r2:{3}'.format(mse_prediction, mae_prediction,
                                                                                   rmse_prediction, r2))

plot_comparison(predictions, test_series)

# %%
model.model.summary()
