from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np


def plot_series(time_series, title: str = 'Time series data'):
    plt.figure(1, figsize=[15, 5])
    plt.plot(time_series)
    plt.xlabel('x')
    plt.xlabel('y')
    plt.title(title)
    plt.show()


def plot_comparison(predicted_data: np.ndarray, real_data: np.ndarray, plot: bool = True,
                    title: str = 'Comparison predicted and real values'):
    """
    Method to plot the predicted and the real values of the time series

    :param predicted_data: predicted data by the neural network
    :param real_data: real data that should has been predicted
    :param plot: flag to show the plot
    :param title: title of the plot
    """
    plt.figure(1, figsize=[15, 5])
    plt.plot(predicted_data, label='Predicted data')
    plt.plot(real_data, label='Real data')
    plt.xlabel('x')
    plt.xlabel('y')
    plt.title(title)
    plt.legend()
    # Save file with timestamp of the execution
    time = datetime.now()
    timestamp = '{0}-{1}-{2}'.format(str(time.hour), str(time.minute), str(time.second))
    plt.savefig('images/comparison_{0}.png'.format(timestamp))
    if plot:
        plt.show()
