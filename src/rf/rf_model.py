"""
AMINA FATMA KHAN
B00868087
March 27, 2026

- Train the Random Forest model

Usage:
    python3 rf_model.py -cls mc
    python3 rf_model.py -cls b
"""

from src.preprocessing.preprocess import *
from src.preprocessing.args import *

import os
import json
import time
import joblib
from sklearn.ensemble import RandomForestClassifier

os.makedirs('saved_models', exist_ok=True)

if __name__ == '__main__':
    args   = args_parser()
    cls    = args.cls
    suffix = 'multiclass' if cls == 'mc' else 'binary'
    print(f'\nTraining RF - {suffix.capitalize()}')

    """
    Data
    """
    data = load_data('data/wsn-ds.csv', cls=cls)
    X_train, X_test, y_train, y_test = split_data(data, test_size=0.3)
    X_train = normalize_data(X_train)
    X_train, y_train = balance_data(X_train, y_train)
    print(f"Training set: {X_train.shape} | Test set: {X_test.shape}")

    """
    Train
    """
    model = RandomForestClassifier(n_estimators=100, random_state=42,
                                   class_weight='balanced', n_jobs=-1)
    start = time.time()
    model.fit(X_train, y_train)
    training_time = round(time.time() - start, 2)
    print(f"Training time: {training_time}s")

    """
    Save model
    """
    joblib.dump(model, f'saved_models/rf_{suffix}_model.pkl')
    print(f"Model saved to saved_models/rf_{suffix}_model.pkl")

    """
    Save history
    """
    with open(f'saved_models/rf_{suffix}_history.json', 'w') as f:
        json.dump({'training_time': training_time}, f, indent=4)
    print(f"History saved to saved_models/rf_{suffix}_history.json")

    print('\nDone!')