"""
Ava Powelson
B00802243
March 16, 2026

Amina Fatma Khan
B00868087
March 27, 2026

Centrally trained MLP model.

See README.md for references.
"""

from src.preprocessing.preprocess import *
from src.preprocessing.args import *
from mlp_model import *

import os
import json
import time

os.makedirs('saved_models', exist_ok=True)

if __name__ == '__main__':
    args = args_parser()
    cls    = args.cls
    suffix = 'multiclass' if cls == 'mc' else 'binary'
    print(f"CLS: {cls}, E: {args.E}, B: {args.B}")

    """
    Load & prepare data
    """
    data = load_data('data/wsn-ds.csv', cls=args.cls)
    x_train, x_test, y_train, y_test = split_data(data)
    x_train = normalize_data(x_train)
    x_test = normalize_data(x_test)
    x_train, y_train = balance_data(x_train, y_train)
    print(f"Training set: {x_train.shape}, Test set: {x_test.shape}")


    """
    Initialize, Train, & Evaluate MLP model
    """
    model = make_mlp(cls=args.cls)
    start = time.time()
    history = train_model(model, x_train, y_train, E=args.E, B=args.B)
    training_time = round(time.time() - start, 2)
    print(f"Training time: {training_time}s")
    results = model.evaluate(x_test, y_test, verbose=0)
    print('Test loss, Test accuracy:', results)

    """
    Save model
    """
    model_path = f'saved_models/mlp_{suffix}_model.keras'
    model.save(model_path)
    print(f"Model saved to {model_path}")

    """
    Save history
    """
    history_path = f'saved_models/mlp_{suffix}_history.json'
    with open(history_path, 'w') as f:
        json.dump({**history.history, 'training_time': training_time}, f, indent=4)
    print(f"History saved to {history_path}")