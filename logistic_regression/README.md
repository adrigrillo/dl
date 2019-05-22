# Deep Learning
## Logistic regression

How to use Logistic Regression

There are two execution modes:

- individual mode: a dataset is selected, specifying the desired features along with
  the learning rate, the weight decay and the number of epochs to train. The logistic regression
  model is trained and then tested, printing the accuracy and the decision boundary.
  To execute the individual mode the following command has to be executed:

    python logistic_regression.py --dataset dataset --epochs epochs --features features -lr learning_rate -rt weight_decay --plot

- bulk mode: a dataset is selected along with a list of learning rates and weight decays. Then
  a set of logistic regression models are trained with all possible binary combinations of features,
  all the learning rates and weights decays. The cost during the training is plotted in this case.
  To execute the bulk mode the following command has to be executed:

    python logistic_regression.py --dataset dataset --epochs epochs --alphas learning_rates --lambdas weight_decays --plot


For example, in the iris dataset using features 0 and 2 without showing the plot:
    python logistic_regression.py --dataset iris --epochs 10000 --features 0 2 -lr 0.2 -rt 0.05

For example, in the monk dataset using features 0 1 3 showing the plot
    python logistic_regression.py --dataset iris --epochs 10000 --features 0 1 3 -lr 0.2 -rt 0.05 --plot

For example, in the iris dataset using a list of learning rates and weight decays:
    python logistic_regression.py --dataset iris --epochs 5000 --alphas 0.005 0.05 0.1 0.3 0.5 0.7 0.9 1 10 --lambdas 0 0.05 0.5 1 10

For example, in the monk dataset using a list of learning rates and weight decays showing the plot
    python logistic_regression.py --dataset monk --epochs 5000 --alphas 0.005 0.05 0.1 0.3 0.5 0.7 0.9 1 10 --lambdas 0 0.05 0.5 1 10 --plot