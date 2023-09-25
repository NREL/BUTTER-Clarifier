"""
BUTTER-Clarifier/test/test_metrics.py
Copyright (c) 2023 Alliance for Sustainable Energy, LLC
License: MIT
Author: Jordan Perr-Sauer
Description: Regression and unit tests for interpretability/metrics.py
"""

import interpretability.metrics as metrics
import tensorflow
import numpy
from functools import cache

GLOBAL_TEST_NET = None

@cache
def _get_basic_sequential(num_layers=4, layer_width=3):
    """
    Returns a basic, pre-trained, sequential, neural network with example inputs and outputs.
    """

    model = tensorflow.keras.Sequential([tensorflow.keras.layers.Dense(layer_width) for i in range(num_layers)])
    model.build(input_shape=(2,2))
    inputs = numpy.random.rand(100,2)
    outputs = numpy.random.rand(100,layer_width)
    
    model.compile(
        # loss='binary_crossentropy', # binary classification
        # loss='categorical_crossentropy', # categorical classification (one hot)
        loss='mean_squared_error',  # regression
        optimizer=tensorflow.keras.optimizers.Adam(0.001),
        # optimizer='rmsprop',
        # metrics=['accuracy'],
    )

    model.fit(
        x=inputs,
        y=outputs,
        epochs=100,
    )

    return (model, inputs, outputs)

### BASIC_STATISTICS

def test_basic_statistics():
    model, inputs, outputs = _get_basic_sequential()
    basic_statistics = metrics.basic_statistics(model)
    assert len(basic_statistics["weight_avg"]) == 4


### Layerwise metrics

def test_cca():
    model, inputs, outputs = _get_basic_sequential()
    result = metrics.cca(model, inputs, outputs)
    assert result.shape == (3,3), "Output shape is not (M-1)x(M-1)"

def test_linear_cka():
    model, inputs, outputs = _get_basic_sequential()
    result = metrics.linear_cka(model, inputs, outputs)
    assert result.shape == (3,3), "Output shape is not (M-1)x(M-1)"


def test_linear_probes():
    model, inputs, outputs = _get_basic_sequential()
    result = metrics.linear_probes(model, inputs, outputs)
    assert len(result["mse"]) == 4, "Length of output is not M"


def test_layerwise_pca():
    model, inputs, outputs = _get_basic_sequential()
    result = metrics.layerwise_pca(model, inputs, outputs)
    assert len(result["explained_variance"]) == 4, "Lebgth of output is not M"


### SPLIT_RELUS

def test_split_relus():
    model, inputs, outputs = _get_basic_sequential()
    split_relus_output = metrics.split_relus(model, inputs)
    assert split_relus_output["split_bool_mask"] != None
