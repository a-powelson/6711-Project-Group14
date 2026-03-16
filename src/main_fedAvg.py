"""
Ava Powelson
B00802243
March 16, 2026

See README.md for references.
"""
from preprocess import *

if __name__ == '__main__':
    data = load_data('../data/wsn-ds.csv')
    data = balance_data(data)

    X_train, X_test, y_train, y_test = split_data(data)
    print(f"Training set: {X_train.shape}, Test set: {X_test.shape}")
