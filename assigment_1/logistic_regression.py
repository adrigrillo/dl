import argparse
import math
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.io import loadmat
from sklearn import preprocessing


def generate_data_with_features(data: pd.DataFrame, features: List[int], elements: int = None,
                                normalise: bool = True, test_elements: int = 0) -> \
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

    :param x: features of the data
    :param y: classes of the data
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


def plot_with_different_rates(train_data: Tuple[pd.DataFrame, pd.DataFrame], learning_rates: List[float],
                              regularization_terms: List[float], epochs: int = 10000,
                              boundary: bool = False, plot: bool = True) -> None:
    """
    Method that trains and plots the cost of a given training data. This method will save the
    figures in the folder 'images' with a name indicating the features used and the regularization
    term used.

    :param train_data: data to train the model, with features and class
    :param learning_rates: list of the learning rates to test
    :param regularization_terms: list of regularization rates to test
    :param epochs: number of epochs to train each combination
    :param boundary: flag to plot the decision boundary
    :param plot: flag to show in the screen the figure
    """
    for rt in regularization_terms:
        plt.figure()
        features_used = train_data[0].T.columns.values
        features_used = "-".join(repr(i) for i in features_used)
        for lr in learning_rates:
            weights, bias, costs = train_model(train_data, epochs=epochs, learning_rate=lr, regularization_term=rt)
            if boundary:
                train_x, train_y = train_data
                str_id = 'f-{0}_lr-{1}_rt{2}'.format(features_used, lr, rt)
                plot_boundary(x=train_x.to_numpy(), y=train_y.to_numpy(), weights=weights,
                              bias=bias, str_id=str_id, plot=plot)
            plt.plot(costs, label='alpha: {0}, lambda: {1}'.format(lr, rt))
            plt.xlabel('Epochs')
            plt.ylabel('Cost')
        plt.title('Training {0} with regularization term {1} and different learning rates'.format(features_used, rt))
        plt.legend()
        plt.savefig('./images/f-{0}_rt-{1}.png'.format(features_used, rt))
        if plot:
            plt.show()


def plot_features_combinations(data: pd.DataFrame, elements: int, learning_rates: List[float],
                               regularization_terms: List[float], epochs: int = 10000,
                               test_elements: int = 0, boundary: bool = False,
                               plot: bool = False) -> None:
    """
    Method that uses all possible binary combination of parameters to train different logistic regressions
    using different learning rates and regularization rates.

    :param data: data to train the model, full dataframe
    :param elements: number of elements that will be used to train
    :param learning_rates: list of the learning rates to test
    :param regularization_terms: list of regularization rates to test
    :param epochs: number of epochs to train each combination
    :param test_elements: number of rows to be used as test
    :param boundary: flag to plot the decision boundary
    :param plot: flag to show in the screen the figure
    """
    num_features = len(data.columns.values) - 1  # remove class
    for i in range(num_features):
        for j in range(i + 1, num_features):
            train, test = generate_data_with_features(data, features=[i, j], elements=elements,
                                                      test_elements=test_elements)
            plot_with_different_rates(train, learning_rates=learning_rates,
                                      regularization_terms=regularization_terms,
                                      epochs=epochs, boundary=boundary, plot=plot)


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


def plot_boundary(x: np.ndarray, y: np.ndarray, weights: np.ndarray,
                  bias: float, str_id: str, plot: bool = True) -> None:
    """
    Method that plot the data and the decision boundary of a given
    logistic regression model, using the bias term and the weights.

    :param x: features of the data
    :param y: classes of the data
    :param weights: weights of the model
    :param bias: bias of the model
    :param plot: flag to show the figure in the screen
    """
    # get the indexes of each class
    zero = np.where(y == 0)[1]
    one = np.where(y == 1)[1]

    plt.scatter(x[0][zero], x[1][zero], s=10, label='Class 0')
    plt.scatter(x[0][one], x[1][one], s=10, label='Class 1')

    # decision boundary
    x_values = [np.min(x[0, :]), np.max(x[0, :])]
    y_values = - (bias + np.dot(weights[0], x_values)) / weights[1]
    plt.plot(x_values, y_values, label='Decision Boundary')

    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend()

    if str_id is None:
        str_id = "-".join(repr(i) for i in weights)

    plt.savefig('./images/boundary-{0}.png'.format(str_id))

    if plot:
        plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset', type=str, choices=['iris', 'monk'], default='iris')
    parser.add_argument('-e', '--epochs', type=int)
    parser.add_argument('-f', '--features', type=int, nargs='+', required=False)
    parser.add_argument('-a', '--alphas', type=float, nargs='+', required=False)
    parser.add_argument('-l', '--lambdas', type=float, nargs='+', required=False)
    parser.add_argument('-b', '--boundary', action='store_true')
    parser.add_argument('-p', '--plot', action='store_true')
    args = parser.parse_args()

    # load datasets
    df = None
    train = None
    test = None
    elements = None
    test_elements = 0

    # generate data
    if args.dataset == 'iris':
        df = pd.read_csv('./data/iris.data', header=None)
        if args.features is None:
            plot_features_combinations(df, elements=100, test_elements=test_elements,
                                       learning_rates=args.alphas, regularization_terms=args.lambdas,
                                       epochs=args.epochs, boundary=args.boundary, plot=args.plot)
        else:
            train, test = generate_data_with_features(df, elements=100, features=args.features)
            weights, bias, costs = train_model(train, epochs=args.epochs, learning_rate=0.2, regularization_term=0.0)
            print(test_model(test, weights, bias))
            test_x, test_y = test
            get_prob_and_cost(test_x.to_numpy(), test_y.to_numpy(), weights, bias)
            plot_boundary(test_x.to_numpy(), test_y.to_numpy(), weights, bias)
    elif args.dataset == "monk":
        df = pd.DataFrame(loadmat('./data/monk2.mat')['monk2'])
        test_elements = math.floor(len(df) * 0.2)
        if args.features is None:
            plot_features_combinations(df, elements=elements, test_elements=test_elements,
                                       learning_rates=args.alphas, regularization_terms=args.lambdas,
                                       epochs=args.epochs,boundary=args.boundary,plot=args.plot)
        else:

            train, test = generate_data_with_features(df, features=args.features, normalise=True,
                                                      test_elements=test_elements)

    # train different rates
    # alphas = [0.005, 0.05, 0.1, 0.3, 0.5, 0.7, 0.9, 1, 10]  # Learning rate
    # lambdas = [0, 0.05, 0.5, 1, 10]
    # plot_features_combinations(df, elements=100, learning_rates=alphas, regularization_terms=lambdas,
    #                            epochs=args.epochs)
    # train
    # weights, bias, costs = train_model(train, epochs=args.epochs, learning_rate=0.2, regularization_term=0.0)
    # print(test_model(test, weights, bias))
    # test_x, test_y = test
    # get_prob_and_cost(test_x.to_numpy(), test_y.to_numpy(), weights, bias)
    # plot_boundary(test_x.to_numpy(), test_y.to_numpy(), weights, bias)
