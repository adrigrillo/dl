#!/usr/bin/env bash

python logistic_regression.py --dataset iris --epochs 10000 --features 0 1 -lr 0.5 -rt 0.05
python logistic_regression.py --dataset iris --epochs 10000 --features 0 2 -lr 0.5 -rt 0.05
python logistic_regression.py --dataset iris --epochs 10000 --features 0 3 -lr 0.5 -rt 0.05
python logistic_regression.py --dataset iris --epochs 10000 --features 1 2 -lr 0.5 -rt 0.05
python logistic_regression.py --dataset iris --epochs 10000 --features 1 3 -lr 0.5 -rt 0.05
python logistic_regression.py --dataset iris --epochs 10000 --features 2 3 -lr 0.5 -rt 0.05