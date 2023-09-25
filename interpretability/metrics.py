"""
BUTTER-Clarifier/interpretability/metrics.py
Copyright (c) 2023 Alliance for Sustainable Energy, LLC
License: MIT
Author: Jordan Perr-Sauer
Description: Functions implementing various techniques to interpret and explain neural networks.
"""

from re import M
import tensorflow as tf
import logging
import numpy as np
from collections import defaultdict

from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, mean_squared_error
#from scikeras.wrappers import KerasRegressor, KerasClassifier

# Basic statistics of the weights and biases

def basic_statistics(model:tf.keras.Sequential):
    """
    Statistics related to basic statsitics on the distirbution of the value of the weights.
    Reference: None, basic
    Output: Dictionay[str,array[float]], where idx of array corresponds to layer number.
    """
    log = defaultdict(list)
    for layer in model.layers:
        weights, biases = layer.get_weights()
        log["weight_avg"].append(float(weights.mean()))
        log["bias_avg"].append(float(biases.mean()))
    return dict(log)

def sparsity(model:tf.keras.Sequential):
    """
    Statistics related to the sparsity of each layer.
    Reference: None, basic
    Output: Dictionay[str,array[float]], where idx of array corresponds to layer number.
    """
    log = defaultdict(list)
    for layer in model.layers:
        weights, biases = layer.get_weights()
        log["nnz_weights"].append( np.sum(np.abs(weights.flatten()) > 0.00001) )
        log["total_weights"].append( weights.flatten().shape[0] )
        log["nnz_biases"].append( np.sum(np.abs(biases.flatten()) > 0.00001) )
        log["total_biases"].append( biases.flatten().shape[0] )
    return dict(log)


# Network dissection using layerwise linear models

def _get_forward_activations(model:tf.keras.Sequential, X_data:np.ndarray):
    """
    Utility function to compute activations of each unit in model with respect to X_data
    """

    logging.info(f"Computing forward activations for model {model}")
    keras_inputs = [model.input]
    keras_outputs = [layer.output for layer in model.layers]
    activations = tf.keras.backend.function(keras_inputs, keras_outputs)([X_data])

    return activations

def layerwise_pca(model:tf.keras.Sequential, inputs, outputs, artifact_path=False):
    """
    Statistics related to the pca of data from each input.
    Reference: Technique inspired by Kornblith, "Similarity of Neural Network Representations Revisited", 2019 and Raghu "SVCCA: Singular Vector Cannonical Correlation Analysis for Deep Learning Dynamics and Interpretability", 2017 
    Inputs:
        - model (keras.Sequential): Keras model
        - inputs (np.array): Array of input data over which to compute the metrics
        - outputs (no.array): Array of outputs corresponding to the inputs
        - artifact_path (str|None): If str, path to directory to save image artifacts
    Output: Dict[str, Array[float]]
    """
    log = defaultdict(list)
    activations = _get_forward_activations(model, inputs)
    for activation in activations:
        pca = PCA()
        pca.fit(activation)
        log["explained_variance"].append(pca.explained_variance_)

    return log

def linear_probes(model:tf.keras.Sequential, inputs, outputs):
    """
    Compute accuracy of linear models trained on the output of each hidden layer.
    Reference: Alain and Bengio, "Understanding intermediate layers using linear classifier probes", 2018
    Output: Dict[str, Array[float]]
    """
    log = defaultdict(list)
    activations = _get_forward_activations(model, inputs)
    for activation in activations:
        lm_outputs = LinearRegression().fit(activation, outputs).predict(activation)
        log["mse"].append(mean_squared_error(outputs, lm_outputs))
        log["mae"].append(mean_absolute_error(outputs, lm_outputs))

    return log


## Layerwise correlation metrics

def linear_cka(model:tf.keras.Model, inputs, outputs):
    """
    Linear Centered Kernel Alignment
    References: Kornblith, "Similarity of Neural Network Representations Revisited", 2019
    Output: MxM np.ndarray(float) where M is the number of layers in the model
    """

    def cka(A,B):
        numerator = np.linalg.norm(A@B.T, ord="fro")**2
        denominator = np.linalg.norm(A@A.T, ord="fro")*np.linalg.norm(B@B.T, ord="fro")
        return 1-(numerator/denominator)

    activations = _get_forward_activations(model, inputs)

    layers = model.layers[:-1] # Last layer is the output layer, with only one neuron, so we must exclude it.

    result = np.ndarray((len(layers), len(layers)), dtype="float")

    for index_a, layer_a in enumerate(layers):
        for index_b, layer_b in enumerate(layers):
            result[index_a, index_b] = cka(activations[index_a],activations[index_b])

    return result

def cca(model:tf.keras.Model, inputs, outputs):
    """
    Linear Centered Kernel Alignment
    References: Kornblith, "Similarity of Neural Network Representations Revisited", 2019
    Output: (M-1)x(M-1) np.ndarray(float) where M is the number of layers in the model
    """
    activations = _get_forward_activations(model, inputs)
    ortho_activations = [np.linalg.qr(X)[0] for X in activations]

    layers = model.layers[:-1] # Last layer is the output layer, with only one neuron, so we must exclude it.

    result = np.ndarray((len(layers), len(layers)), dtype="float")

    for index_a, layer_a in enumerate(layers):
        for index_b, layer_b in enumerate(layers):

            qx = ortho_activations[index_a]
            qy = ortho_activations[index_b]
            features_x = activations[index_a]
            features_y = activations[index_b]

            cca_ab = np.linalg.norm(qx.T.dot(qy)) ** 2 / min(
                features_x.shape[1], features_y.shape[1])

            result[index_a, index_b] = cca_ab

    return result


## RELU-Speciffic Metrics

def split_relus(model:tf.keras.models.Sequential, X_data, outputs=None):
    """
    Compute which Relu neurons are "split" by the input data.
    Note, only use this function on sequential, dense models with RELU activation function.
    References: Split neurons are mentioned in Bak, "nnenum: Verification of ReLU Neural Networks with Optimized Abstraction Refinement", 2021
    Inputs:
        - model:tf.keras.models.Sequential with dense layers of RELU Activations
        - inputs:np.ndarray matching the expected input shape of model.
    Outputs:
        - out:dict Dictionary with:
            - split_bool_mask:list[np.ndarray<bool>]: List over each layer showing which neurons are split within each layer.
            - split_count_below:list[np.ndarray<int>]: Same, but counts the number of examples falling below the elbow.
            - split_ratio:list[np.ndarray<float>]: Same, but divides by the number of examples.
    """
    assert type(model) == tf.keras.models.Sequential, "split_relus metric only works on sequential models for now"
    log = defaultdict(list)
    for layer in model.layers:
        if type(layer) != tf.keras.layers.Dense:
            # logger.debug(f"Skipping layer {layer}")
            continue
        ## Compute the input at each layer. TODO: Might be more efficient to compute this in one shot for all layers.
        probe_fn = tf.keras.backend.function([model.input], [layer.input])
        inputs = probe_fn([X_data])[0]
        ## Manually compute each step of this Dense layer. These lines more or less follow what's in keras.layer.Dense
        pre_bias = tf.matmul(a=inputs, b=layer.kernel)
        pre_activation = tf.nn.bias_add(pre_bias, layer.bias)
        post_activation = layer.activation(pre_activation)
        ## Compute Splits
        split_bool_mask = ~tf.math.reduce_all(pre_activation < 0, axis=0)
        split_count_below = tf.math.reduce_sum(tf.cast(pre_activation < 0, dtype=tf.int32), axis=0)
        split_ratio = split_count_below / X_data.shape[0]
        ## Package the output
        log["split_bool_mask"].append(split_bool_mask)
        log["split_count_below"].append(split_count_below)
        log["split_ratio"].append(split_ratio)
    return log


## Global Sensitivity

# def permutation_feature_importance(model:tf.keras.Model, inputs, outputs):
#     pass

# def partial_dependence_plots():
#     pass

# def prototypes_criticisms():
#     pass

# def ice_plots():
#     pass

## Output / Class based metrics? CAVs?

## Output / Generative metrics?

## Compute All Metrics
# def compute_all(model, inputs, outputs, artifact_path=None):
#     pass