"""
BUTTER-Clarifier/interpretability/callbacks.py
Copyright (c) 2023 Alliance for Sustainable Energy, LLC
License: MIT
Author: Jordan Perr-Sauer
Description: Keras API callback to integrate interpretability module into Keras training routines.
"""

import interpretability.metrics as metrics
from interpretability.utils import NpEncoder
import tensorflow
import json
from collections import defaultdict

from pathlib import Path

class InterpretabilityMetricsKerasCallback(tensorflow.keras.callbacks.Callback):

    def __init__(self, epochs, data, metrics="all", save_to_path=False, save_to_history=True):
        """
        Keras Callback
        """
        self.epochs = epochs
        self.save_to_path = save_to_path
        self.data = data
        
        if metrics == "all":
            self.metrics = ["sparsity",
                       "basic_statistics",
                       "layerwise_pca",
                       "linear_probes",
                       "cca",
                       "split_relu"]
        elif type(metrics) == str:
            self.metrics = [metrics]
        else:
            self.metrics = metrics

        self.save_to_history = save_to_history
        self.history = defaultdict(list)

    def compute_the_metrics(self, logs=None):
        """
        Compute all metrics we are able to compute, and return the results as a dictionary.
        """

        X_train, X_test, y_train, y_test = self.data
        
        out = {}

        ## Call all of the metric functions needed, and store their output as entries in out
        
        if "sparsity" in self.metrics:
            out["sparsity"] = metrics.sparsity(self.model)
        
        if "basic_statistics" in self.metrics:
            out["basic_statistics"] = metrics.basic_statistics(self.model)
        
        if "layerwise_pca" in self.metrics:
            out["layerwise_pca"] = metrics.layerwise_pca(self.model, X_test, y_test)
        
        if "linear_probes" in self.metrics:
            out["linear_probes"] = metrics.linear_probes(self.model, X_test, y_test)
        
        if "split_relus" in self.metrics:
            out["split_relus"] = metrics.split_relus(self.model, X_train)
        
        if "cca" in self.metrics:
            out["cca"] = metrics.cca(self.model, X_test, y_test)

        ## Store pass-thru logs from caller
        out["logs"] = logs
        
        return out

    def on_epoch_end(self, epoch, logs=None):
        """
        Keras callback API: on_epoch_end
        """

        if epoch % self.epochs == 0:

            out = self.compute_the_metrics()

            if self.save_to_path:
                base_path = self.save_to_path+f"/epoch={epoch}"
                Path(base_path).mkdir(parents=True, exist_ok=True)
                with open(base_path+'/logs.json', 'w') as outfile:
                    json.dump(out, outfile, cls=NpEncoder)
            
            if self.save_to_history:
                for key, value in out.items():
                    self.history[key].append(value)
                self.history["epoch"].append(epoch)