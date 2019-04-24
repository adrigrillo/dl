#!/usr/bin/env bash

python logistic_regression.py --dataset monk --epochs 5000 --alphas 0.005 0.05 0.1 0.3 0.5 0.8 1.0 --lambdas 0.0 0.05 0.2 0.5 1.0 5.0
