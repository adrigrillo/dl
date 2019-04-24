#!/usr/bin/env bash

python logistic_regression.py --dataset monk --epochs 10000 --features 0 1 -lr 0.5 -rt 0.05
python logistic_regression.py --dataset monk --epochs 10000 --features 0 2 -lr 0.5 -rt 0.05
python logistic_regression.py --dataset monk --epochs 10000 --features 0 3 -lr 0.5 -rt 0.05
python logistic_regression.py --dataset monk --epochs 10000 --features 0 4 -lr 0.5 -rt 0.05
python logistic_regression.py --dataset monk --epochs 10000 --features 0 5 -lr 0.5 -rt 0.05
python logistic_regression.py --dataset monk --epochs 10000 --features 1 2 -lr 0.5 -rt 0.05
python logistic_regression.py --dataset monk --epochs 10000 --features 1 3 -lr 0.5 -rt 0.05
python logistic_regression.py --dataset monk --epochs 10000 --features 1 4 -lr 0.5 -rt 0.05
python logistic_regression.py --dataset monk --epochs 10000 --features 1 5 -lr 0.5 -rt 0.05
python logistic_regression.py --dataset monk --epochs 10000 --features 2 3 -lr 0.5 -rt 0.05
python logistic_regression.py --dataset monk --epochs 10000 --features 2 4 -lr 0.5 -rt 0.05
python logistic_regression.py --dataset monk --epochs 10000 --features 2 5 -lr 0.5 -rt 0.05
python logistic_regression.py --dataset monk --epochs 10000 --features 3 4 -lr 0.5 -rt 0.05
python logistic_regression.py --dataset monk --epochs 10000 --features 3 5 -lr 0.5 -rt 0.05
python logistic_regression.py --dataset monk --epochs 10000 --features 4 5 -lr 0.5 -rt 0.05