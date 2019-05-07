from typing import Tuple

import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import KFold

min_max_scaler = preprocessing.MinMaxScaler()


def normalize_data(time_series: np.ndarray) -> np.ndarray:
    """
    Normalization of the data

    :param time_series: time series to normalize
    :return: normalized time series
    """
    time_series = np.reshape(time_series, newshape=(time_series.shape[0], 1))
    return min_max_scaler.fit_transform(time_series)


def process_data(data: np.ndarray, x_window_size: int, y_window_size: int = 1,
                 dimensions: int = 2) -> Tuple[np.ndarray, np.ndarray]:
    """
    Method that process a time series and creates the data required to train and test the
    neural network.

    For example, for a x windows of size 3 and y, 1:

    [1, 2, 3, 4, 5] ->  [[1, 2, 3], [2, 3, 4]]; [[4], [5]]

    :param data: time series data
    :param x_window_size: number of data points that are included in the x elements
    :param y_window_size: number of data points that are included in the y elements
    :param dimensions: dimensionality of the data, 2D or 3D depending of the network
    :return: tuple with the values of x and y and the specified shapes
    """
    n_possible_elements = data.shape[0] - x_window_size - y_window_size
    if dimensions == 2:
        shape_x = (x_window_size,)
        shape_y = (y_window_size,)
    else:
        shape_x = (x_window_size, 1)
        shape_y = (y_window_size, 1)
    x = np.empty((n_possible_elements,) + shape_x)
    y = np.empty((n_possible_elements,) + shape_y)
    for i in range(n_possible_elements):
        x_end = i + x_window_size
        y_end = x_end + y_window_size
        x_element = np.reshape(data[i:x_end], newshape=shape_x)
        y_element = np.reshape(data[x_end:y_end], newshape=shape_y)
        x[i] = x_element
        y[i] = y_element
    return x, y


def generate_train_and_validation_sets(x_data: np.ndarray, y_data: np.ndarray, folds: int,
                                       seed: int = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Method that splits the data in training and test using k-folds.

    :param x_data: inputs
    :param y_data: value to predict outputs
    :param folds: number of data divisions
    :param seed: seed for the selection of the instances
    :return: arrays with the train/test x and y instances
    """
    k_folder = KFold(n_splits=folds, random_state=seed)
    x_train_folds = list()
    y_train_folds = list()
    x_test_folds = list()
    y_test_folds = list()
    for train_index, test_index in k_folder.split(x_data):
        x_train, x_test = x_data[train_index], x_data[test_index]
        y_train, y_test = y_data[train_index], y_data[test_index]
        x_train_folds.append(x_train)
        x_test_folds.append(x_test)
        y_train_folds.append(y_train)
        y_test_folds.append(y_test)
    return np.array(x_train_folds), np.array(y_train_folds), np.array(x_test_folds), np.array(y_test_folds)
