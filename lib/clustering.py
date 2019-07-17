from collections import defaultdict

import numpy as np
import pandas as pd
import sklearn.cluster

import os
import subprocess as sp

from modulecontainers import Module, Modules
from simdist import simdist

def standardize(X):
    return (X - X.mean())/(X.std())

def dummy(E, n=10, **kwargs):
    labels = np.random.randint(0, n, len(E.columns))
    modules = convert_labels2modules(labels, E.columns)
    return modules

def agglom(E, k=100, linkage="complete", simdist_function="pearson_correlation", **kwargs):
    distances = simdist(E, simdist_function, similarity = False)
    agglom = sklearn.cluster.AgglomerativeClustering(n_clusters=int(k), affinity = "precomputed", linkage = linkage)
    agglom.fit(distances)
    modules = convert_labels2modules(agglom.labels_, E.columns)
    return modules

## utility functions
def convert_labels2modules(labels, G, ignore_label=None):
    modules = defaultdict(Module)
    for label, gene in zip(labels, G):
        if label != ignore_label:
            modules[label].add(gene)
    return list(modules.values())

def convert_modules2labels(modules, G):
    labels = {}
    for i, module in enumerate(modules):
        for g in module:
            labels[g] = i
    return labels
