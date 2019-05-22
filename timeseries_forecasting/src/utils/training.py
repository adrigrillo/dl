import numpy as np
from keras import backend
from sklearn.metrics import mean_squared_error


def root_mean_square_error(y_true, y_pred):
    return backend.sqrt(backend.mean(backend.square(y_pred - y_true), axis=-1))


def calculate_rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


def normalised_mean_square_error(y_true, y_pred):
    return backend.mean(backend.square(y_pred - y_true), axis=-1) / (np.std(y_true) ** 2)
