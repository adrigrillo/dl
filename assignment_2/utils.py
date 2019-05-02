"""Contains functions to generate artificial data for predictions as well as
a function to plot predictions."""
from typing import Tuple

import numpy as np


def process_data(time_series: np.ndarray, window_size: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Method that process a time series and creates the data required to train and test the
    neural network.

    For example, for a windows of size 3:

    [1, 2, 3, 4, 5] ->  [[1, 2, 3], [2, 3, 4]]; [[4], [5]]

    :param time_series: time series data
    :param window_size: number of data points that are included in the x elements
    :return: tuple with the values of x and y
    """
    x = []
    y = []
    # number of examples that can be created with the given time series
    num_possible_x = time_series.shape[0] - window_size
    for x_first_index in range(num_possible_x - 1):
        x_last_index = x_first_index + window_size
        x.append(time_series[x_first_index:x_last_index])
        y.append(time_series[x_last_index])
    return np.array(x), np.array(y)
