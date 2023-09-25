"""
BUTTER-Clarifier/test/test_callback.py
Copyright (c) 2023 Alliance for Sustainable Energy, LLC
License: MIT
Author: Jordan Perr-Sauer
Description: Regression and unit tests for interpretability/callbacks.py
"""

import tensorflow
import numpy as np
from interpretability.callbacks import InterpretabilityMetricsKerasCallback

def _get_basic_sequential():
    model = tensorflow.keras.Sequential([tensorflow.keras.layers.Dense(3) for i in range(4)])
    model.build(input_shape=(2,2))
    return model

def test_callback_with_internal_history():
    model = _get_basic_sequential()
    X_train = np.random.random((2,2))
    X_test = np.random.random((2,2))
    Y_train = np.random.random((2,3))
    Y_test = np.random.random((2,3))
    data = (X_train, X_test, Y_train, Y_test)

    callback = InterpretabilityMetricsKerasCallback(10,data)

    model.compile(
        # loss='binary_crossentropy', # binary classification
        # loss='categorical_crossentropy', # categorical classification (one hot)
        loss='mean_squared_error',  # regression
        optimizer=tensorflow.keras.optimizers.Adam(0.001),
        # optimizer='rmsprop',
        # metrics=['accuracy'],
    )

    model.fit(
        x=X_train,
        y=Y_train,
        epochs=100,
        callbacks=[
            callback
        ]
    )

    assert "epoch" in callback.history.keys(), "History does not contain an epoch key."
    assert len(callback.history["epoch"]) == 10, "History has incorrect number of epochs"
