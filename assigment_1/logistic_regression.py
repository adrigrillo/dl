import argparse
import math
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.io import loadmat
from sklearn import preprocessing


def generate_data_with_features(data: pd.DataFrame, features: List[int], elements: int = None,
                                normalise: bool = False, test_elements: int = 0) -> \
        Tuple[Tuple[pd.DataFrame, pd.DataFrame], Tuple[pd.DataFrame, pd.DataFrame]]:
    """
    Method to obtain a dataset with the features and the size desired. This
    method return the data and the class that belongs to it. If test data is desired
    the method will return two different tuples, both containing a dataframe with the
    features and other with the class.

    To retrieve correctly the data of the method, the following examples should be
    followed:

    ```
    train, test = generate_data_with_features(...)
    ```

    or

    ```
    train_x, train_y, test_x, test_y = generate_data_with_features(...)
    ```

    Normalisation are made in column by column style, using only numeric columns.

    :param data: dataframe with the original data
    :param features: list with the column name of the features
    :param elements: number of elements, when not specified the full dataset will be used
    :param normalise: flag to normalise the data
    :param test_elements: int with the number of rows to be used as test
    :return: dataframes with the new data with the desired characteristics
    """
    if elements is None:  # take full data
        elements = len(data)

    train_x = data[features][:elements]
    train_y = data.iloc[:, -1][:elements]

    if normalise:
        for i in features:
            column = train_x[i].values.reshape(-1, 1)
            if column.dtype == np.float64 or column.dtype == np.int64:
                min_max_scaler = preprocessing.MinMaxScaler()
                scaled_column = min_max_scaler.fit_transform(column)
                train_x[i] = scaled_column.reshape(-1)

    le = preprocessing.LabelEncoder()  # transform the class to 0 and 1
    train_y = le.fit_transform(train_y)

    test_x = None
    test_y = None
    if test_elements != 0 and elements > test_elements:
        train_elements = elements - test_elements
        test_x = train_x[train_elements:]
        train_x = train_x[:train_elements]
        test_y = train_y[train_elements:]
        train_y = train_y[:train_elements]

    return (pd.DataFrame(train_x).T, pd.DataFrame(train_y).T), (pd.DataFrame(test_x).T, pd.DataFrame(test_y).T)


def sigmoid(z: np.ndarray) -> np.ndarray:
    """
    The sigmoid function that applies to the result of the product of the weight and the activation of the
    neurons plus the biases, known as weighted input.
    z = w_l*a_l+b

    :param z: weighted input.
    :return: activation of the next layer of the network
    """
    return 1.0 / (1 + np.exp(-z))


def forward_pass(x: np.ndarray, weights: np.ndarray, bias: float) -> np.ndarray:
    """
    Method that calculates the activation of the sigmoid function
    from a given input, weights and biases.

    :param x: input data
    :param weights: tuple with the weights
    :param bias: value of the biases
    :return: activation of the model for each row of the input
    """
    weighted_input = np.dot(weights, x) + bias
    return sigmoid(weighted_input)


def calculate_derivatives(x: np.ndarray, y: np.ndarray, weights: np.ndarray, bias: float,
                          regularization_term: float = 0) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    Method that propagates the input and calculates the cost and the derivative of the
    weights and the biases.

    :param x: input data
    :param y: output data
    :param weights: weights of the model
    :param bias: bias of the model
    :param regularization_term: value of lambda
    :return: tuple with the cost, the derivative of the weights and bias
    """
    n_samples = y.shape[1]
    activation = forward_pass(x, weights, bias)
    cost = np.mean(-y * np.log(activation) - (1 - y) * np.log(1 - activation))
    cost = cost + regularization_term / (2 * n_samples) * np.dot(weights.T, weights)  # lambda/2m*sum(theta^2)
    dz = -(y - activation) / n_samples
    dw = np.dot(dz, x.T).squeeze()
    db = np.sum(dz)
    return cost, dw, db


def train_model(train_data: Tuple[pd.DataFrame, pd.DataFrame], epochs: int, learning_rate: float = 0.5,
                regularization_term: float = 0) -> Tuple[np.ndarray, float, np.ndarray]:
    """
    Method to train the model

    :param train_data: training data, with the features and the outputs
    :param epochs: number of epochs to train the model
    :param learning_rate: value of the learning rate, alpha
    :param regularization_term: value of the regularization term, lambda
    :return: weights and bias of the trained model. List of the costs during training
    """
    x, y = train_data  # extract the data and the classes
    n_samples = y.shape[1]
    costs = list()
    bias = 1
    weights = np.random.uniform(low=-0.7, high=0.7, size=x.shape[0])
    for epoch in range(epochs):
        cost, dw, db = calculate_derivatives(x=x.to_numpy(), y=y.to_numpy(), weights=weights,
                                             bias=bias, regularization_term=regularization_term)
        if epoch % 500 == 0:
            print('The cost in epoch {0} was {1}'.format(epoch, cost))
        costs.append(cost)
        weights -= learning_rate * (dw + regularization_term / n_samples * weights)
        bias -= learning_rate * (db + regularization_term / n_samples * bias)
    print('Finished training, trained during {0}'.format(epochs))
    return weights, bias, np.array(costs)


def predict(x: np.ndarray, weights: np.ndarray, bias: float) -> np.ndarray:
    """
    Method that outputs the system response
    of a given input, weights and biases.

    :param x: input data
    :param weights: tuple with the weights
    :param bias: value of the biases
    :return: output of the system for each row of the input
    """
    activation = forward_pass(x, weights, bias)
    return 1 * (activation > 0.5)


def get_prob_and_cost(x: np.ndarray, y: np.ndarray, weights: np.ndarray,
                      bias: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Method that outputs the system response
    of a given input, weights and biases.

    :param x: input data
    :param weights: tuple with the weights
    :param bias: value of the biases
    :return: output of the system
    """
    activation: np.ndarray = forward_pass(x, weights, bias)
    cost: np.ndarray = -y * np.log(activation) - (1 - y) * np.log(1 - activation)
    return activation, cost.squeeze()


def test_model(test_data: Tuple[pd.DataFrame, pd.DataFrame], weights: np.ndarray,
               bias: float) -> float:
    """
    Method that calculates the percentage of error given a test dataset.

    :param test_data: test data, with the features and the outputs
    :param weights: weights of the trained model
    :param bias: bias of the trained model
    :return:
    """
    x, y = test_data
    predicted_y = predict(x.to_numpy(), weights, bias)
    diff_pred_real = abs(predicted_y - y.to_numpy().squeeze())
    percentage_error = np.count_nonzero(diff_pred_real == 1) / len(diff_pred_real)
    return percentage_error


def plot_boundary(x: np.ndarray, y: np.ndarray, weights: np.ndarray, bias: float):
    # Get the indexes of each class
    zero = np.where(y == 0)[1]
    one = np.where(y == 1)[1]
    # plots
    plt.scatter(x[0][zero], x[1][zero], s=10, label='Class 0')
    plt.scatter(x[0][one], x[1][one], s=10, label='Class 1')

    x_values = [np.min(x[0, :]), np.max(x[0, :])]
    y_values = - (bias + np.dot(weights[0], x_values)) / weights[1]

    plt.plot(x_values, y_values, label='Decision Boundary')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset', type=str, choices=['iris', 'monk'], default='iris')
    parser.add_argument('-f', '--features', type=int, nargs='+')
    parser.add_argument('-e', '--epochs', type=int)
    args = parser.parse_args()

    # load datasets
    df = None
    train = None
    test = None
    test_elements = 0

    # generate data
    if args.dataset == 'iris':
        df = pd.read_csv('./data/iris.data', header=None)
        train, test = generate_data_with_features(df, elements=100, features=args.features, normalise=True)
    elif args.dataset == "monk":
        df = pd.DataFrame(loadmat('./data/monk2.mat')['monk2'])
        test_elements = math.floor(len(df) * 0.2)
        train, test = generate_data_with_features(df, features=args.features, normalise=True,
                                                  test_elements=test_elements)

    # train
    weights, bias, costs = train_model(train, epochs=args.epochs, learning_rate=0.2, regularization_term=0.0)
    print(test_model(train, weights, bias))
    get_prob_and_cost(train[0].to_numpy(), train[1].to_numpy(), weights, bias)
    plot_boundary(train[0].to_numpy(), train[1].to_numpy(), weights, bias)
