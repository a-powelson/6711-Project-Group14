"""
Ava Powelson
B00802243
March 16, 2026

See README.md for references.
"""
from preprocess import *
from mlp_model import *
import numpy as np

if __name__ == '__main__':
    """
    Initialize global model
    """
    gbl_model = make_mlp()

    """
    Load & prepare data
    """
    gbl_data = load_data('data/wsn-ds.csv')

    client_data = []
    client_models = []
    for i in range(0, 100):
        client_data.append(gbl_data[gbl_data['id'] % 100 == i])
        client_models.append(make_mlp())

    """
    Select Clients (using all for now)
    """
    C = [x for x in range(0, 100)]
    
    """
    Do FedAvg
    """
    for i in range(0, 3):
        agg_weights = [np.zeros_like(w) for w in gbl_model.get_weights()]
        total_samples = 0
        for k in C:
            x_train, x_test, y_train, y_test = split_data(client_data[k])
            x_train = normalize_data(x_train)
            x_test = normalize_data(x_test)

            train_model(client_models[k], x_train, y_train)

            wk = client_models[k].get_weights()
            sample_size_k = len(x_train)

            for i in range(len(agg_weights)):
                agg_weights[i] += wk[i] * sample_size_k
            total_samples += sample_size_k

        # Average weights
        agg_weights = [w / total_samples for w in agg_weights]

        # Update global model
        gbl_model.set_weights(agg_weights)

        # Update clients
        for k in C:
            client_models[k].set_weights(gbl_model.get_weights())

    """
    Evaluate final global model
    """
    x_train, x_test, y_train, y_test = split_data(gbl_data)
    x_test = normalize_data(x_test)
    res = gbl_model.evaluate(x_test, y_test)
    print('Test loss, Test accuracy:', res)
