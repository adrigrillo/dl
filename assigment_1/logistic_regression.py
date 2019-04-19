from typing import List

import numpy as np
import pandas as pd
from sklearn import preprocessing


def generate_data_with_features(data: pd.DataFrame, features: List[int], elements: int = 100,
                                normalise: bool = False) -> pd.DataFrame:
    """
    Method to obtain a dataset with the features and the size desired. This
    methods uses sampling without replacement to obtain the elements.

    The indices are reset to make the sample independent of the previous dataset.
    Normalisation are made in column by column style, using only numeric columns.

    :param data: dataframe with the original data
    :param features: list with the column name of the features
    :param elements: number of elements
    :param normalise: boolean to normalise the data
    :return: dataframe with the new data with the desired characteristics
    """
    data = data[features].sample(elements, random_state=47).reset_index(drop=True)
    if normalise:
        for i in features:
            column = data[i].values.reshape(-1, 1)
            if column.dtype == np.float64 or column.dtype == np.int64:
                min_max_scaler = preprocessing.MinMaxScaler()
                scaled_column = min_max_scaler.fit_transform(column)
                data[i] = scaled_column.reshape(-1)
    return data


def sigmoid(z):
    """
    The sigmoid function that applies to the result of the product of the weight and the activation of the
    neurons plus the biases, known as weighted input.
    z = w_l*a_l+b

    :param z: weighted input.
    :return: activation of the next layer of the network
    """
    return 1.0 / (1 + np.exp(-z))


def sigmoid_derivative(activation_values):
    """
    Derivative of the sigmoid function

    :param activation_values: activation values
    :return: result of applying the derivative function
    """
    return activation_values * (1.0 - activation_values)


def predict():



if __name__ == '__main__':
    df = pd.read_csv('./iris.data', header=None)
    x = generate_data_with_features(df, [0, 2, 4], normalise=True)
    print(x.head())
