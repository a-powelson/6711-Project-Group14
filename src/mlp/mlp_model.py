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
from args import DEFAULT_E, DEFAULT_B, DEFAULT_CLS

def make_mlp(cls=DEFAULT_CLS):

    # Multi class
    if cls == 'mc':
        model = Sequential([
            Input(shape=(18,)),
            Dense(256, activation='relu'),  
            Dense(128, activation='relu'), 
            Dense(5, activation='softmax'), # output layer
        ])

    # Binary classification
    else:
        model = Sequential([
            Input(shape=(18,)),
            Dense(256, activation='relu'),  
            Dense(128, activation='relu'), 
            Dense(1, activation='softmax'), # output layer
        ])

    """
    Metrics ToDo: find names that tf uses for other ones
    """
    model.compile(optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])

    return model

def train_model(model, x_train, y_train, E=DEFAULT_E, B=DEFAULT_B):
    model.fit(x_train, y_train, epochs=E, 
        batch_size=B, 
        validation_split=0.2)
    
    return model