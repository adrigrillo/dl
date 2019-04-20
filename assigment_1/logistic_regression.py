from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn import preprocessing


def generate_data_with_features(data: pd.DataFrame, features: List[int], elements: int = 100, normalise: bool = False,
                                randomise: bool = False) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Method to obtain a dataset with the features and the size desired. This
    method return the data and the class that belongs to it.

    When randomise is true, the function uses sampling without replacement to obtain the elements.
    The indices are reset to make the sample independent of the previous dataset.
    Normalisation are made in column by column style, using only numeric columns.

    :param data: dataframe with the original data
    :param features: list with the column name of the features
    :param elements: number of elements
    :param normalise: flag to normalise the data
    :param randomise: flag to randomise the selection
    :return: dataframe with the new data with the desired characteristics
    """
    if randomise is False:
        new_x = data[features][:elements]
        new_y = data.iloc[:, -1][:elements]
    else:
        new_x = data[features].sample(elements, random_state=47).reset_index(drop=True)
        new_y = data.iloc[:, -1].sample(elements, random_state=47).reset_index(drop=True)
    if normalise:
        for i in features:
            column = new_x[i].values.reshape(-1, 1)
            if column.dtype == np.float64 or column.dtype == np.int64:
                min_max_scaler = preprocessing.MinMaxScaler()
                scaled_column = min_max_scaler.fit_transform(column)
                new_x[i] = scaled_column.reshape(-1)
    le = preprocessing.LabelEncoder()  # transform the output to 0 and classes - 1
    new_y = le.fit_transform(new_y)
    return pd.DataFrame(new_x).T, pd.DataFrame(new_y).T


def sigmoid(z):
    """
    The sigmoid function that applies to the result of the product of the weight and the activation of the
    neurons plus the biases, known as weighted input.
    z = w_l*a_l+b

    :param z: weighted input.
    :return: activation of the next layer of the network
    """
    return 1.0 / (1 + np.exp(-z))


def predict(x: np.ndarray, weights: np.ndarray, bias: float):
    """
    Method that calculates the output from a given input, weights and biases.

    :param x: input data
    :param weights: tuple with the weights
    :param bias: value of the biases
    :return: output of the system
    """
    weighted_input = np.dot(weights, x) + bias
    return sigmoid(weighted_input)


def calculate_derivatives(x: np.ndarray, y: np.ndarray, weights: np.ndarray, bias: float):
    """
    Method that propagates the input and calculates the cost and the derivative of the
    weights and the biases.

    :param x: input data
    :param y: output data
    :param weights: weights of the model
    :param bias: bias of the model
    :return: tuple with the cost, the derivative of the weights and bias
    """
    n_samples = y.shape[1]
    activation = predict(x, weights, bias)
    cost = np.mean(-y * np.log(activation) - (1 - y) * np.log(1 - activation))
    dz = -(y - activation) / n_samples
    dw = np.dot(dz, x.T)
    db = np.sum(dz)
    return cost, dw, db


def train_model(train_data: Tuple[pd.DataFrame, pd.DataFrame], learning_rate: float, epochs: int):
    x, y = train_data  # extract the data and the classes
    bias = 1
    weights = np.random.uniform(low=-0.7, high=0.7, size=x.shape[0])
    for epoch in range(epochs):
        cost, dw, db = calculate_derivatives(x.values, y.values, weights, bias)
        weights += learning_rate * dw
        bias += learning_rate * bias
    return weights, bias


if __name__ == '__main__':
    df = pd.read_csv('./iris.data', header=None)
    train = generate_data_with_features(df, [0, 2], normalise=True)
    train_model(train, train, 0.2, 100)
