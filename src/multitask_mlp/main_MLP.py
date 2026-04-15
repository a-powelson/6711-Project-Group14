"""
Ava Powelson
B00802243
April 15, 2026

Centrally trained MLP model.

See README.md for references.
"""
from preprocess import *
from mlp_model import *
from localize import get_coords
from args import *

if __name__ == '__main__':
    args = args_parser()
    cls = args.cls

    """
    Load & prepare data
    """
    data = load_data('data/wsn-ds.csv', cls=args.cls)
    
    # Add estimated locations
    x_coords, y_coords = get_coords(data)
    x_col = []
    y_col = []

    for id in data['id'] % 100:
        x_col.append(x_coords[id])
        y_col.append(y_coords[id])

    data['x'] = x_col
    data['y'] = y_col

    x_train, x_test, y_train_class, y_test_class, y_train_loc, y_test_loc = split_data_loc(data)
    x_train = normalize_data(x_train)
    x_test = normalize_data(x_test)
    # x_train, y_train_class = balance_data(x_train, y_train_class)
    print(f"Training set: {x_train.shape}, Test set: {x_test.shape}")

    """
    Initialize, Train, & Evaluate MLP model
    """
    model = make_mlp(cls=args.cls)
    train_model(model, x_train, y_train_loc, y_train_class)
    results = model.evaluate(x_test, {'loc_output': y_test_loc, 
                                      'class_output': y_test_class}, 
                             return_dict=True, verbose=2)
    print(results)
