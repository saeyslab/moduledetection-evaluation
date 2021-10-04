from modulecontainers import Bicluster, Modules, Module

from sklearn.cluster import SpectralBiclustering 

def standardize(X):
    return (X - X.mean(0))/(X.std(0))

def spectral_biclust(E, ngenes=3, nconditions=1,  spectral_method="bistochastic", n=6, n_best_ratio=0.5, **kwargs):
    """
    Note:
    - method was moved from sklearn.cluster.bicluster.SpectralBiclustering

    """
    n_best = max([int(n*n_best_ratio), 1])

    spectral = SpectralBiclustering(n_clusters=(nconditions,ngenes), method=spectral_method, n_components=n, n_best=n_best)

    spectral.fit(standardize(E))

    bics = []
    for columns, rows in zip(spectral.columns_, spectral.rows_):
        genes = E.columns[columns]
        conditions = E.index[rows]

        bics.append(Bicluster(genes, conditions))

    return bics
    