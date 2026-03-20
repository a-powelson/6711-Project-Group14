"""
Ava Powelson
B00802243
March 16, 2026

See README.md for references.

Tuneable characteristics:
- number of layers
- size of layers
- optimizer
- loss function
- batch size
- epochs
"""
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input

def make_mlp():
    model = Sequential([
        Input(shape=(18,)),
        Dense(256, activation='relu'),  
        Dense(128, activation='relu'), 
        Dense(5, activation='softmax'), # output layer
    ])

    """
    Metrics ToDo: find names that tf uses for other ones
    """
    model.compile(optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])

    return model

def train_model(model, x_train, y_train):
    model.fit(x_train, y_train, epochs=10, 
        batch_size=256, 
        validation_split=0.2)
    
    return model