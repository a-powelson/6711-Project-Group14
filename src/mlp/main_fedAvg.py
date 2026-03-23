"""
Ava Powelson
B00802243
March 16, 2026

See README.md for references.
"""
from preprocess import *
from mlp_model import *
from args import *
import numpy as np
import random

if __name__ == '__main__':
    """
    Load args
    """
    args = args_parser()
    C = args.C
    T = args.T
    E = args.E
    B = args.B
    print(f"CLS: {args.cls}, T: {T}, C: {C}, E: {E}, B: {B}")

    """
    Initialize global model & select clients
    """
    gbl_model = make_mlp(cls=args.cls)
    client_ids = random.sample(range(0, 100), C)
    
    """
    Load & prepare data
    """
    gbl_data = load_data('data/wsn-ds.csv', cls=args.cls)

    client_data = []
    client_models = []
    for i in range(0, C):
        client_data.append(gbl_data[gbl_data['id'] % 100 == client_ids[i]])
        client_models.append(make_mlp(cls=args.cls))
        client_models[i].set_weights(gbl_model.get_weights())
    
    """
    Do FedAvg
    """
    for i in range(0, T):
        agg_weights = [np.zeros_like(w) for w in gbl_model.get_weights()]
        total_samples = 0
        for k in range(0, C):
            x_train, x_test, y_train, y_test = split_data(client_data[k])
            x_train = normalize_data(x_train)
            x_test = normalize_data(x_test)

            train_model(client_models[k], x_train, y_train, E=E, B=B)

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
        for k in range(0, C):
            client_models[k].set_weights(gbl_model.get_weights())

    """
    Evaluate final global model
    """
    x_train, x_test, y_train, y_test = split_data(gbl_data)
    x_test = normalize_data(x_test)
    res = gbl_model.evaluate(x_test, y_test)
    print('Test loss, Test accuracy:', res)
