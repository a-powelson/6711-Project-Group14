"""
Ava Powelson
B00802243
March 16, 2026

Amina Fatma Khan
B00868087
March 27, 2026

See README.md for references.
"""

from src.preprocessing.preprocess import *
from src.preprocessing.args import *
from mlp_model import *
import numpy as np
import random
import os
import json
import time

os.makedirs('saved_models', exist_ok=True)

if __name__ == '__main__':
    """
    Load args
    """
    args = args_parser()
    C = args.C
    T = args.T
    E = args.E
    B = args.B
    suffix = 'multiclass' if args.cls == 'mc' else 'binary'
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
    Global test set for evaluation after training and evaluate file
    """
    x_train_gbl, x_test_gbl, y_train_gbl, y_test_gbl = split_data(gbl_data)
    x_test_gbl = normalize_data(x_test_gbl)

    client_data = []
    client_models = []
    for i in range(0, C):
        client_data.append(gbl_data[gbl_data['id'] % 100 == client_ids[i]])
        client_models.append(make_mlp(cls=args.cls))
        client_models[i].set_weights(gbl_model.get_weights())

    """
    Do FedAvg
    """
    # Per-round metrics for evaluate file
    round_accs     = []
    round_losses   = []
    avg_val_accs   = []
    avg_val_losses = []

    start = time.time()

    for i in range(0, T):
        agg_weights = [np.zeros_like(w) for w in gbl_model.get_weights()]
        total_samples = 0
        client_val_accs   = []
        client_val_losses = []

        for k in range(0, C):
            x_train, x_test, y_train, y_test = split_data(client_data[k])
            x_train = normalize_data(x_train)
            x_test = normalize_data(x_test)

            hist = train_model(client_models[k], x_train, y_train, E=E, B=B)
            client_val_accs.append(hist.history['val_accuracy'][-1])
            client_val_losses.append(hist.history['val_loss'][-1])

            wk = client_models[k].get_weights()
            sample_size_k = len(x_train)

            for j in range(len(agg_weights)):
                agg_weights[j] += wk[j] * sample_size_k
            total_samples += sample_size_k

        # Average weights
        agg_weights = [w / total_samples for w in agg_weights]

        # Update global model
        gbl_model.set_weights(agg_weights)

        # Update clients
        for k in range(0, C):
            client_models[k].set_weights(gbl_model.get_weights())

        # Evaluate global model on global test set
        loss, acc = gbl_model.evaluate(x_test_gbl, y_test_gbl, verbose=0)
        round_losses.append(loss)
        round_accs.append(acc)
        avg_val_accs.append(float(np.mean(client_val_accs)))
        avg_val_losses.append(float(np.mean(client_val_losses)))
        print(f'Global - loss: {loss:.4f}  acc: {acc:.4f}')

    training_time = round(time.time() - start, 2)
    print(f"\nTotal training time: {training_time}s") 

    """
    Evaluate final global model
    """
    x_train, x_test, y_train, y_test = split_data(gbl_data)
    x_test = normalize_data(x_test)
    res = gbl_model.evaluate(x_test, y_test)
    print('Test loss, Test accuracy:', res)

    """
    Save model
    """
    model_path = f'saved_models/fedavg_{suffix}_model.keras'
    gbl_model.save(model_path)
    print(f"Model saved to {model_path}")

    """
    Save history
    """
    history = {
        'round_accs':     round_accs,
        'round_losses':   round_losses,
        'avg_val_accs':   avg_val_accs,
        'avg_val_losses': avg_val_losses,
        'training_time':  training_time,
    }
    history_path = f'saved_models/fedavg_{suffix}_history.json'
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=4)
    print(f"History saved to {history_path}")