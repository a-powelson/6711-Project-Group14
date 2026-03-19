"""
Ava Powelson
B00802243
March 16, 2026

See README.md for references.
"""
from preprocess import *
from mlp_model import *

if __name__ == '__main__':
    """
    Load & prepare data
    """
    data = load_data('data/wsn-ds.csv')

    x_train, x_test, y_train, y_test = split_data(data)
    x_train = normalize_data(x_train)
    x_test = normalize_data(x_test)

    x_train, y_train = balance_data(x_train, y_train)
    print(f"Training set: {x_train.shape}, Test set: {x_test.shape}")

    """
    Initialize global model
    """
    gbl_model = make_mlp()

    """
    Do FedAvg...
    0. For total rounds T:
        1. Sample N clients to form C
        2. For each client k:
                set weights = global weights

    """

    """
    Evaluate final global model
    """