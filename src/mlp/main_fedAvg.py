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
    Do FedAvg
    """
    for i in range(0, 3):
        # Select clients C
        # Provide models for all clients k in C
        # Train local clients
        for i, k in C:
            train_model(client_model, client_x, client_y)

        # Aggregate client models
        agg_model = make_mlp(...)
        for k in C:
            agg_model_weights += k_model_weights

        # Update global model
        gbl_model_weights = agg_model_weights / len(C)

    """
    Evaluate final global model
    """
    res = gbl_model.evaluate(...)
    print('Test loss, Test accuracy:', res)
    