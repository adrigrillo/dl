"""Contains functions to generate artificial data for predictions as well as
a function to plot predictions."""
from typing import Tuple

import numpy as np
from sklearn.model_selection import KFold


def process_data(time_series: np.ndarray, window_size: int, shape_x: Tuple = (3, 1),
                 shape_y: Tuple = (1, 1)) -> Tuple[np.ndarray, np.ndarray]:
    """
    Method that process a time series and creates the data required to train and test the
    neural network.

    For example, for a windows of size 3:

    [1, 2, 3, 4, 5] ->  [[1, 2, 3], [2, 3, 4]]; [[4], [5]]

    :param time_series: time series data
    :param window_size: number of data points that are included in the x elements
    :param shape_x: shape of the elements of x. Default = (3, 1)
    :param shape_y: shape of the elements of y. Default = (1, 1)
    :return: tuple with the values of x and y and the specified shapes
    """
    # number of examples that can be created with the given time series
    n_possible_elements = time_series.shape[0] - window_size - 1
    x = np.empty((n_possible_elements,) + shape_x)
    y = np.empty((n_possible_elements,) + shape_y)
    for i in range(n_possible_elements):
        end = i + window_size
        x_element = np.reshape(time_series[i:end], newshape=shape_x)
        y_element = np.reshape(time_series[end], newshape=shape_y)
        x[i] = x_element
        y[i] = y_element
    return np.array(x), np.array(y)


def generate_train_and_validation_sets(x_data: np.ndarray, y_data: np.ndarray, folds: int,
                                       seed: int = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
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
