"""
Ava Powelson
B00802243
March 16, 2026

See README.md for references.
"""
from preprocess import *
from mlp_model import *

if __name__ == '__main__':
    data = load_data('data/wsn-ds.csv')
    data = balance_data(data)

    x_train, x_test, y_train, y_test = split_data(data)
    print(f"Training set: {x_train.shape}, Test set: {x_test.shape}")

    model = make_mlp(x_train, y_train)

    results = model.evaluate(x_test, y_test, verbose=0)
    print('Test loss, Test accuracy:', results)