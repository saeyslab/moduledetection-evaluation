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

def ica_zscore(E, k=200, stdcutoff=1e-3, seed=None, **kwargs):
    source = _ica_fastica(E, k, seed)
    modules = _ica_zscore(E, source, stdcutoff)

    return modules

def _ica_fastica(E, k, seed=None):
    ica = sklearn.decomposition.FastICA(n_components=int(k), random_state=seed)
    source = ica.fit_transform(standardize(E).T)

    return source

def _ica_zscore(E, source, stdcutoff):
    modules = []
    for source_row in source.T:
        genes = E.columns[source_row < -source_row.std() * stdcutoff].tolist() + E.columns[source_row > +source_row.std() * stdcutoff].tolist()

        modules.append(Module(genes))
    return modules

def meanshift(E, bandwidth=None, cluster_all=True, **kwargs):
    if bandwidth is None or bandwidth == "auto":
        meanshift = sklearn.cluster.MeanShift(cluster_all=cluster_all)
    else:
        meanshift = sklearn.cluster.MeanShift(bandwidth=bandwidth, cluster_all=cluster_all)

    meanshift.fit(standardize(E).T)
    meanshift.labels_

    modules = convert_labels2modules(meanshift.labels_, E.columns)

    return modules

def baseline_permuted(modules, **kwargs):
    modules = Modules(modules)
    modules = modules.shuffle()
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
