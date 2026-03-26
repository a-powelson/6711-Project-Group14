"""
Ava Powelson
B00802243
March 26, 2026

See README.md for references.
"""
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input
from args import DEFAULT_E, DEFAULT_B, DEFAULT_CLS

"""
Make a multi-task MLP with two heads to simultaneously
perform classification and localization tasks.
Cannot use Sequential anymore as it is more suited for 
Single-Input/Single-Output, so switching to Keras's
Functional API.
"""
def make_mlp(cls=DEFAULT_CLS):
    # Shared layers
    inputs = Input(shape=(18,))
    l1 = Dense(256, activation='relu')(inputs)
    l2 = Dense(128, activation='relu')(l1)
    l3 = Dense(64, activation='relu')(l2)

    # Head 1: Location
    loc_head = Dense(64, activation='relu')(l3)
    loc_output = Dense(2, activation='linear', name='loc_output')(loc_head)

    # Head 2: Classification
    class_head = Dense(64, activation='relu')(l3)
    if cls == 'mc':  # multi-class
        class_output = Dense(5, activation='softmax', name='class_output')(class_head)
    else:  # binary
        class_output = Dense(1, activation='sigmoid', name='class_output')(class_head)

    # Model
    model = Model(inputs=inputs, outputs=[loc_output, class_output])
    model.compile(optimizer='adam', 
        loss={'loc_output': 'mse', 'class_output':
                'sparse_categorical_crossentropy' },
        metrics={ 'loc_output': ['mae'],
                'class_output': ['accuracy'] }
    )

    return model

def train_model(model, x_train, y_train_loc, y_train_class, E=DEFAULT_E, B=DEFAULT_B):
    model.fit(x_train, { 'loc_output': y_train_loc,
                         'class_output': y_train_class
                       }, epochs=E, 
                        batch_size=B, 
                        validation_split=0.2)
    
    return model