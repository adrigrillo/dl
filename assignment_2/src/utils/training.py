import numpy as np
from keras import backend


def root_mean_square_error(y_true, y_pred):
    return backend.sqrt(backend.mean(backend.square(y_pred - y_true), axis=-1))


def normalised_mean_square_error(y_true, y_pred):
    return backend.mean(backend.square(y_pred - y_true), axis=-1) / (np.std(y_true) ** 2)
