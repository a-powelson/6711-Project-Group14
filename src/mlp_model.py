"""
Ava Powelson
B00802243
March 16, 2026

See README.md for references.

Tuneable characteristics:
- activation function
- number of layers
- size of layers
- optimizer
- loss function
- batch size
"""
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense

def make_mlp():
    model = Sequential([
        Dense(256, activation='relu'),  
        Dense(128, activation='relu'), 
        Dense(5, activation='softmax'), # output layer
    ])

    """
    Adam: a popular optimizer that extends SGD
    Loss: suitable for multi-class classification
    Metrics: accuracy, plus we will likely add more. 
        - I'm not sure if they get added here or later -AP
    """
    model.compile(optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])

    return model

def train_model(model, x_train, y_train):
    """
    In the GeeksForGeeks MLP tutorial they split the data, and then use only
    20% of the training portion, I'm not sure why. Need to look into that 
    more. - AP
    """
    model.fit(x_train, y_train, epochs=10, 
        batch_size=2000, 
        validation_split=0.2)
    
    return model