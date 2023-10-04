# BUTTER-Clarifier

![Tests Workflow Badge](https://github.com/jordanperr/BUTTER-Clarifier/actions/workflows/tests.yml/badge.svg)

This repository contains a python package of neural network interpretability techniques (`interpretability`) and a keras callback to easily compute and capture data related to these techniques (we call these values `metrics`) during training. It was developed to be used with the [BUTTER Deep Learning Experimental Framework](https://github.com/NREL/BUTTER-Empirical-Deep-Learning-Experimental-Framework), but does not depend on this framework and may be useful to projects outside of this framework.

The vision for this codebase is to collect algorithms for explainable artificial intelliegnce (XAI) in a single framework that is easy to use, easy to read, and can be expand upon. Here, we package XAI algorithms into a module called "metrics", which are implemented as python functions. The return type of a metric is typically a dictionary holding data of multiple data types, such as real values and numpy matrices. Callbacks and any other connector code is provided as necessary in a separate module to make these metrics more easily usable. This project depends on Tensorflow's Keras API, although it would be nice to try and support multiple backends one day.

## Installation

### Dependencies
- Developed using Python 3.11 with pip on Linux.
- Tensorflow 2.13+
- Scikit-Learn

### VSCode Devcontainer

This package was developed using VSCode Dev Containers and contains VSCode configuration files. The easiest way to get started developing this package is to check this project out from Github using VSCode and open it using the VSCode Dev Containers extension.

### Manual Install using Pip

Manually installing the package is also easy. To manually clone and install this project:

```shell
$ git clone [URL TO THIS REPO]
$ cd ./BUTTER-Clarifier
$ pip install -e .[test]
```

### Testing

Pytest should be installed automatically when using the `[test]` extra argument, but you can install it manually using `pip install pytest` if needed. Tests can be run with the following command:

```shell
$ pytest ./test
```

## Usage

### Using the Keras callback
For training with Keras, a convenient Keras callback class is provided. The callback will compute the explainability metrics during model training and can store the results either in memory or as JSON files. The signature of its constructor is:

`InterpretabilityMetricsKerasCallback(epochs, data, metrics="all", save_to_path=False, save_to_history=True)`

- epochs: Record metrics every multiple of this epoch.
- data: 4-tuple with (X_test, X_train, y_test, y_train)
- metrics: "all" is the only supported option for now, and is the default.
- save_to_path: If string, save the metrics to this path as JSON files.
- save_to_history: If true, save the metrics in memory using the callback's history property.

Example of how to use the callback during training:
```python
from interpretability.callbacks import InterpretabilityMetricsKerasCallback

X_test, X_train, y_test, y_train = ...
model = ...

interpretability_callback = InterpretabilityMetricsKerasCallback(20, (X_train, X_test, y_train, y_test))

model.fit(
    ...
    callbacks=[
            interpretability_callback
        ]
)

print(interpretability_callback.history)

```

### Using the metrics module

Instead of using the Keras callback, you can also use metric functions directly. Each set of metrics are implemented as Python functions. They have similar signatures, with the model being the first argument. However, some metrics require extra inputs and so they do not have idential signatures.

Example:
```python
from interpretability.metrics import basic_statistics, linear_probes

X_test, X_train, y_test, y_train = ...
model = ...

# Simplest metric
out = basic_statistics(model)
print(out)

# Some metrics require data
out = linear_probes(model, X_test, y_test)
print(out)
```

### List of Metrics

The following metrics are implemented. For more information on these metrics, including academic citations, please see the inline documentation.

| metric name | description | outputs |
| ---- | ---- | ---- |
| basic_statistics | Package of basic statistics of weights and biases | Means of the weights and biases | 
| sparsity | Package of metrics relateded to sparsity | Number of non-zero weights and biases. |
| layerwise_pca | Computes principal component analysis (PCA) of activations for each layer | Explained variance | 
| linear_probes | Computes a linear regression of activations for each layer | MSE and MAE error for each layer | 
| linear_cka | Linear Centered Kernel Alignment | Correlation matrix over the number of hidden layers | 
| cca | Cannonical Correlation Analysis | Correlation matrix over the number of hidden layers | 
| split_relus | Computes those hidden units which are split by the given input data | Boolean mask over hidden units, as well as total count and ratio of split units | 


## Repository Contents

- /interpretability: Python package containing metrics and the keras callback.
    - Callbacks.py: Classes conforming to Keras callback API.
    - Metrics.py: Functions that implement the explainability metrics.
- /test: Python tests that can be run from the command line or as part of a CI pipeline.

## No Expectation of Support

This software is released without the expectation of support. If you run into issues using this software, you may report an issue using the Github issue tracker. However, the developers do not make any guarantees about responding to these issues.

## Contributing and Roadmap

Some places where this codebase could be improved include:
- Adding more metrics.
- Adding more comprehensive tests.
- Improving the documentation and generating a documentation website.

Contributions are welcome, but we can not guarantee responsiveness around pull requests. We recommend first creating an issue in the Github issue tracker to alert us of your intent to contribute. Then, fork this repository into your own account and make the changes in your own fork. Finally, submit a pull request into our `develop` branch from your fork referencing the issue.

## Acknowledgements

This software was written by Jordan Perr-Sauer as part of the National Renewable Energy Laboratory's Laboratory Directed Research and Development program. It is released under software record SWR-23-61 "DMP Interpretability"