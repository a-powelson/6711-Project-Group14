"""
Ava Powelson
B00802243
March 16, 2026

See README.md for references.

Notes from GeeksForGeeks:
Advantages
    Versatility: MLPs can be applied to a variety of problems, both classification and regression.
    Non-linearity: Using activation functions MLPs can model complex, non-linear relationships in data.
    Parallel Computation: With the help of GPUs, MLPs can be trained quickly by takfing advantage of parallel computing.

Disadvantages
    Computationally Expensive: MLPs can be slow to train especially on large datasets with many layers.
    Prone to Overfitting: Without proper regularization techniques they can overfit the training data leading to poor generalization.
    Sensitivity to Data Scaling: They require properly normalized or scaled data for optimal performance.
"""
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense

def make_mlp(x_train, y_train):
    """
    I'm not sure I fully understand these parameters, it seems we'll likely
    have to change them to suit our data -AP
    """
    model = Sequential([
        Flatten(),
        Dense(256, activation='sigmoid'),  
        Dense(128, activation='sigmoid'), 
        Dense(10, activation='softmax'),  
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

    """
    In the GeeksForGeeks MLP tutorial they split the data, and then use only
    20% of the training portion, I'm not sure why. Need to look into that 
    more. - AP
    """
    mod = model.fit(x_train, y_train, epochs=10, 
            batch_size=2000, 
            validation_split=0.2)
    
    print(mod)

    return mod
