import numpy as np
import pandas as pd
from copy import deepcopy

from modulecontainers import Module, Modules

from sklearn.metrics.pairwise import euclidean_distances

def optimalmodules(modulesets, distances, autoprocedure = "asw"):
    if autoprocedure == "asw":
        asws = [asw(modules, distances) for modules in modulesets]
        print(asws)
        return modulesets[np.nanargmax(asws)], np.nanargmax(asws)
        #return modulesets[np.argmax(asws)], np.argmax(asws)
    elif autoprocedure == "ptbiserial":
        ptbiserial = [asw(modules, distances) for modules in modulesets]
        print(ptbiserial)
        return modulesets[np.nanargmax(ptbiserial)], np.nanargmax(ptbiserial)
        #return modulesets[np.argmax(asws)], np.argmax(asws)


def cal_cvi(modules, E, distances, scorename, clustercenters=None, dispersions=None):
    if scorename == "asw":
        return asw(modules, distances)
    elif scorename == "ptbiserial":
        return ptbiserial(modules, distances)
    elif scorename == "dbstarindex":
        return dbstarindex(modules, distances, clustercenters, dispersions)
    elif scorename == "dbindex":
        return dbindex(modules, E, clustercenters, dispersions)
    elif scorename == "ch":
        return ch(modules, E)
    elif scorename == "sf":
        return sf(modules, E)

def cal_clustercenters(modules, E):
    clustercenters = []
    for module in modules:
        clustercenters.append(E.ix[:,module].mean(1))
    clustercenters = np.array(clustercenters).T

    if any(pd.isnull(clustercenters.flatten())):
        print(modules)

    return clustercenters

def cal_dispersions(modules, E, clustercenters):
    dispersions = []
    for center, module in zip(clustercenters.T, modules):
        if len(module) > 1:
            dispersion = np.sqrt(((E.ix[:, module].T - center) ** 2).sum(1)).mean()
        else:
            dispersion = 0
        dispersions.append(dispersion)
    dispersions = np.array(dispersions)

    return dispersions

def sf(modules, E, clustercenters=None, dispersions=None):
    # related to the ch index

    if len(modules) < 2:
        return np.inf

    W = np.zeros((len(E.index), len(E.index)))
    for module in modules:
        if len(module) < 2:
            covar = 0
        else:
            covar = np.cov(E.ix[:,module])
        W = W + ((len(module) - 1) * covar)

    S = (len(E.columns) - 1) * np.cov(E)
    B = S - W

    bcd = np.diag(B).sum() / (len(E.columns) * len(modules))

    if clustercenters is None:
        clustercenters = cal_clustercenters(modules, E)
    if dispersions is None:
        dispersions = cal_dispersions(modules, E, clustercenters)

    wcd = dispersions.sum()

    return 1-(1/(bcd-wcd))


def ch(modules, E):
    # based on the R fpc implementation
    if len(modules) < 2:
        return np.inf

    W = np.zeros((len(E.index), len(E.index)))
    for module in modules:
        if len(module) < 2:
            covar = 0
        else:
            covar = np.cov(E.ix[:,module])
        W = W + ((len(module) - 1) * covar)

    S = (len(E.columns) - 1) * np.cov(E)
    B = S - W

    numer = np.diag(B).sum()
    denom = np.diag(W).sum()

    return (len(E.columns) - len(modules))/(len(modules) - 1) * numer/denom


def dbindex(modules, E, clustercenters=None, dispersions=None):
    if len(modules) < 2:
        return np.inf

    if clustercenters is None:
        clustercenters = cal_clustercenters(modules, E)

    centerdistances = euclidean_distances(clustercenters.T)

    if dispersions is None:
        dispersions = cal_dispersions(modules, E, clustercenters)

    score = 0
    for moduleid in range(len(modules)):
        othermoduleids = [moduleid2 for moduleid2 in range(len(modules)) if moduleid2 != moduleid]
        otherdispersions = dispersions[othermoduleids]
        score += np.max((dispersions[moduleid] + otherdispersions) / centerdistances[moduleid, othermoduleids])

    score = score / len(modules)

    return score

def dbstarindex(modules, E, clustercenters=None, dispersions=None):
    if len(modules) < 2:
        return np.inf

    if clustercenters is None:
        clustercenters = cal_clustercenters(modules, E)

    centerdistances = euclidean_distances(clustercenters.T)

    if dispersions is None:
        dispersions = cal_dispersions(modules, E, clustercenters)

    score = 0
    for moduleid in range(len(modules)):
        othermoduleids = [moduleid2 for moduleid2 in range(len(modules)) if moduleid2 != moduleid]
        otherdispersions = dispersions[othermoduleids]
        score += np.max((dispersions[moduleid] + otherdispersions)) / np.min(centerdistances[moduleid, othermoduleids])

    score = score / len(modules)

    return score


def ptbiserial(modules, distances):
    if len(modules) < 2:
        return -np.inf

    connectivity = modules.cal_connectivity(distances.columns).astype(np.float)
    distances = distances.copy()

    np.fill_diagonal(connectivity.values, np.nan)
    np.fill_diagonal(distances.values, np.nan)

    connectivity = connectivity.as_matrix().flatten()
    distances = distances.as_matrix().flatten()

    connectivity = 1-connectivity[~np.isnan(connectivity)]
    distances = distances[~np.isnan(distances)]

    return np.corrcoef(connectivity, distances)[0,1]

def asw(modules, distances):
    distances = distances.copy()

    # make sure every gene is in at least one module
    # otherwise, this measure is EXTREMELY biased towards modulesets with only a handful of genes into very compact modules
    # if (modules.cal_membership(distances.columns).sum(1) == 0).sum() > 0:
    #     modules = deepcopy(modules)
    #     modules.append(Module(distances.columns[modules.cal_membership(distances.columns).sum(1) == 0]))

    if len(modules) < 2:
        return -np.inf

    averagemoduledistances = []
    np.fill_diagonal(distances.values, np.nan)
    for module in modules:
        averagemoduledistances.append(distances.ix[module].mean(skipna=True)) # skipna is here a little trick to ignore the distance of a gene with himself
    averagemoduledistances = pd.DataFrame(averagemoduledistances)

    asw_modules = []
    for i, module in enumerate(modules):

        if len(module) <= 1:
            asw_module = [0]
        else:
            C = averagemoduledistances.ix[:,module]
            A = C.ix[i]
            B = C.drop(i).min()
            
            asw_module = ((B-A)/np.max([B, A], 0))
        asw_modules.extend(asw_module)
    return np.array(asw_modules).mean()
def asw_modules(modules, distances):
    distances = distances.copy()
    np.fill_diagonal(distances.values, np.nan)

    # make sure every gene is in at least one module
    # otherwise, this measure is EXTREMELY biased towards modulesets with only a handful of genes into very compact modules
    # if (modules.cal_membership(distances.columns).sum(1) == 0).sum() > 0:
    #     modules = deepcopy(modules)
    #     modules.append(Module(distances.columns[modules.cal_membership(distances.columns).sum(1) == 0]))

    averagemoduledistances = []
    for module in modules:
        averagemoduledistances.append(distances.ix[module].mean(skipna=True)) # skipna is here a little trick to ignore the distance of a gene with himself
    averagemoduledistances = pd.DataFrame(averagemoduledistances)

    asw_modules = []
    for i, module in enumerate(modules):
        if len(module) <= 1:
            asw_module = [0]
        else:
            C = averagemoduledistances.ix[:,module]
            A = C.ix[i]
            B = C.drop(i).min()
            
            asw_module = ((B-A)/np.max([B, A], 0))
        asw_modules.append(asw_module.mean())
    return asw_modules


def cal_module_average(modules, E):
    Emod = []
    for module in modules:
        Emod.append(E[list(module)].mean(1))
    Emod = pd.DataFrame(Emod, columns=E.index).T
    return Emod