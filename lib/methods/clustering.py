from collections import defaultdict

import numpy as np
import pandas as pd
import sklearn.cluster

import os
import subprocess as sp

from modulecontainers import Module, Modules
from simdist import simdist
from util import TemporaryDirectory

try:
    import nimfa
except ImportError:
    "No nimfa"

try:
    import rpy2.robjects as ro
    import rpy2.robjects.numpy2ri
    import rpy2.robjects.pandas2ri
    from rpy2.robjects.packages import importr
    rpy2.robjects.numpy2ri.activate()
    rpy2.robjects.pandas2ri.activate()
except ImportError:
    "No rpy2"


def standardize(X):
    return (X - X.mean())/(X.std())

def dummy(E, n=10, **kwargs):
    labels = np.random.randint(0, n, len(E.columns))
    modules = convert_labels2modules(labels, E.columns)
    return modules

## clustering functions
def flame(E, knn=10,threshold=-1, threshold2=-3.0, steps=500, **kwargs):
    with TemporaryDirectory() as tmpdir:
        with open(tmpdir + "/E.csv", "w") as outfile:
            outfile.write(str(E.shape[1]) + " " + str(E.shape[0]) + "\n")
            standardize(E).T.to_csv(outfile, index=False, header=False, sep=" ")

        binary = os.environ["PERSOFTWARELOCATION"] + "/flame/sample"
        command = "{binary} {tmpdir}/E.csv {knn} {threshold2} {steps} {threshold}".format(**locals())

        process = sp.Popen(command, shell=True, stdout=sp.PIPE)
        out, err = process.communicate()

        modules = []
        for row in out.decode().split("\n"):
            if row.startswith("Cluster") and "outliers" not in row:
                gids = row[row.index(":")+1:].split(",")
                if gids[0] != "":
                    module = Module([E.columns[int(gid)] for gid in gids])
                    modules.append(module)

    return modules

def kmedoids(E, k=100, simdist_function="pearson_correlation", **kwargs):
    importr("cluster")

    distances = simdist(E, simdist_function, similarity=False, **kwargs)
    rresults = ro.r["pam"](distances, diss=True, k=k)
    modules = convert_labels2modules(list(rresults.rx2("clustering")), E.columns)
    return modules

def som(E, dim=6, dims=None, topo="rectangular", rlen=100, alpha=[0.05, 0.01], radius=None, **kwargs):
    importr("kohonen")

    if dims is None:
        dims = [dim, dim]

    ro.globalenv["E"] = (standardize(E).T)

    if radius is None:
        rresults = ro.r["som"](ro.r["as.matrix"](standardize(E).T), ro.r["somgrid"](dims[0], dims[1], "rectangular"), rlen=rlen, alpha=alpha)
    else:
        rresults = ro.r["som"](ro.r["as.matrix"](standardize(E).T), ro.r["somgrid"](dims[0], dims[1], "rectangular"), rlen=rlen, alpha=alpha, radius=radius)

    modules = convert_labels2modules(list(rresults.rx2("unit.classif")), E.columns)

    return modules

def kmeans(E, k=100, max_iter=300, n_init=10, seed=None, **kwargs):
    kmeans = sklearn.cluster.KMeans(n_clusters=int(k), max_iter=int(max_iter), n_init=int(n_init), random_state=seed)
    kmeans.fit(standardize(E).T)
    kmeans.fit(E.T)
    modules = convert_labels2modules(kmeans.labels_, E.columns)
    return modules

def cmeans(E, k=100, m="auto", cutoff=0.5, cluster_all=True, **kwargs):
    importr("Mfuzz")
    importr("Biobase")
    Exprs = ro.r["ExpressionSet"](ro.r["as.matrix"](standardize(E).T))
    if m == "auto":
        m = ro.r["mestimate"](Exprs)

    rresults = ro.r["mfuzz"](Exprs, k, m)

    membership = np.array(rresults.rx2("membership"))

    modules = []
    for membership_cluster in membership.T:
        genes = E.columns[membership_cluster >= cutoff]
        modules.append(Module(genes))

    return modules

def spectral_similarity(E, k=100, seed=None, simdist_function="pearson_correlation", **kwargs):
    similarities = simdist(E, simdist_function, **kwargs)
    spectral = sklearn.cluster.SpectralClustering(n_clusters=int(k), affinity="precomputed", random_state = seed)
    spectral.fit(similarities+1)

    return convert_labels2modules(spectral.labels_, E.columns)

def affinity(E, preference_fraction=0.5, simdist_function="pearson_correlation", damping=0.5, max_iter=200, **kwargs):
    similarities = simdist(E, simdist_function, **kwargs)

    similarities_max, similarities_min = similarities.as_matrix().max(), similarities.as_matrix().min()
    preference = (similarities_max - similarities_min) * preference_fraction

    ro.packages.importr("apcluster")

    rresults = ro.r["apcluster"](s=ro.Matrix(similarities.as_matrix()), p=preference)
    labels = np.array(ro.r["labels"](rresults, "enum"))

    modules = convert_labels2modules(labels, E.columns)

    return modules

def spectral_knn(E, k=100, knn=50, seed=None, **kwargs):
    spectral = sklearn.cluster.SpectralClustering(n_clusters=int(k), n_neighbors = int(knn), affinity="nearest_neighbors", random_state = seed)
    spectral.fit(standardize(E).T)

    return convert_labels2modules(spectral.labels_, E.columns)

def wgcna(E, power=6, mergeCutHeight=0.15, minModuleSize=20, deepSplit=2, detectCutHeight=0.995, TOMDenom="min", reassignThreshold=1e-6, **kwargs):
    importr("WGCNA")

    ro.r("allowWGCNAThreads()")

    rblockwiseModules = ro.r["blockwiseModules"]
    rresults = rblockwiseModules(
        E,
        power=power,
        mergeCutHeight=mergeCutHeight,
        minModuleSize=minModuleSize,
        deepSplit=deepSplit,
        detectCutHeight=detectCutHeight,
        numericLabels=True,
        TOMDenom=TOMDenom,
        reassignThreshold=reassignThreshold
    )

    modules = convert_labels2modules(list(rresults.rx2("colors")), E.columns, ignore_label=0)

    return modules

def agglom(E, k=100, linkage="complete", simdist_function="pearson_correlation", **kwargs):
    importr("cluster")
    ro.globalenv["distances"] =  simdist(E, simdist_function, similarity=False, **kwargs)
    ro.r("hclust_results = hclust(as.dist(distances), method='{linkage}')".format(**locals()))
    rresults = ro.r("labels = cutree(hclust_results, k={k})".format(**locals()))
    modules = convert_labels2modules(list(rresults), E.columns)
    return modules

def hybrid(E, k=100, **kwargs):
    importr("hybridHclust")

    ro.globalenv["E"] = standardize(E).T
    ro.r("hclust_results = hybridHclust(E)")
    rresults = ro.r("cutree(hclust_results, k={k})".format(**locals()))

    modules = convert_labels2modules(list(rresults), E.columns)

    return modules

def divisive(E, k=100, **kwargs):
    importr("cluster")

    ro.globalenv["E"] = E
    ro.r("diana_results = diana(as.dist(1-cor(E)),diss=TRUE)".format(**locals()))
    rresults = ro.r("cutree(diana_results, k={k})".format(**locals()))

    modules = convert_labels2modules(list(rresults), E.columns)

    return modules

def sota(E, maxCycles=1000, maxEpochs=1000, distance="euclidean", wcell=0.01, pcell=0.005, scell=0.001, delta=1e-04, neighb_level=0, alpha=0.95, unrest_growth=False, **kwargs):
    importr("clValid")

    distances = simdist(standardize(E), "euclidean", False, **kwargs)
    maxDiversity = np.percentile(distances.as_matrix().flatten(), alpha)

    rresults = ro.r["sota"](standardize(E).T.as_matrix(), maxCycles, maxEpochs, distance, wcell, pcell, scell, delta, neighb_level, maxDiversity, unrest_growth)

    modules = convert_labels2modules(list(rresults.rx2("clust")), E.columns)

    return modules

def dclust(E, rho=0.5, delta=0.5, simdist_function="pearson_correlation", **kwargs):
    ro.packages.importr("densityClust")

    distances = simdist(E, simdist_function, False, **kwargs)
    rresults =  ro.r["densityClust"](ro.r["as.dist"](distances))
    rresults = ro.r["findClusters"](rresults, rho=rho, delta=delta)

    modules = convert_labels2modules(list(rresults.rx2("clusters")), E.columns)

    return modules

def click(E, homogeneity=0.5, **kwargs):
    with TemporaryDirectory() as tmpdir:
        with open(tmpdir + "/clickInput.orig", "w") as outfile:
            outfile.write("{nG} {nC}\n".format(nG = len(E.columns), nC=len(E.index)))

            E.T.to_csv(outfile, sep="\t", header=False)

        with open(tmpdir + "/clickParams.txt", "w") as outfile:
            outfile.write("""DATA_TYPE
FP 
INPUT_FILES_PREFIX
{tmpdir}/clickInput 
OUTPUT_FILE_PREFIX
{tmpdir}/clickOutput 
SIMILARITY_TYPE
CORRELATION 
HOMOGENEITY
{homogeneity}
            """.format(tmpdir=tmpdir, homogeneity=homogeneity))

        click_location = os.environ["PERSOFTWARELOCATION"] + "/Expander/click.exe"

        command = "{click_location} {tmpdir}/clickParams.txt".format(**locals())

        sp.call(command, shell=True)

        labels = pd.read_csv(tmpdir + "/clickOutput.res.sol", sep="\t", index_col=0, header=None, squeeze=True)

    modules = convert_labels2modules(labels.tolist(), labels.index.tolist(), 0)
    return modules

def dbscan(E, eps=0.2, MinPts=5, **kwargs):
    importr("fpc")

    ro.globalenv["E"] = E
    rresults = ro.r("dbscan(as.dist(1-cor(E)), method='dist', eps={eps}, MinPts={MinPts})".format(**locals()))
    modules = convert_labels2modules(list(rresults.rx2("cluster")), E.columns, 0)

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

def clues(E, disMethod="1-corr", n0=300, alpha=0.05, eps=1e-4, itmax=20, strengthMethod="sil", strengthIni=-1, **kwargs):
    ro.packages.importr("clues")

    rresults = ro.r["clues"](
        ro.Matrix(standardize(E).T.as_matrix()),
        disMethod=disMethod,
        n0=n0,
        alpha=alpha,
        eps=eps,
        itmax=itmax,
        strengthMethod=strengthMethod,
        strengthIni=strengthIni,
        quiet=False
    )

    modules = convert_labels2modules(list(rresults.rx2("mem")), E.columns)

    return modules

def transitivity(E, threshold=0.1, simdist_function="pearson_correlation", cutoff=-1, **kwargs):
    similarities = simdist(E, simdist_function, **kwargs)

    with TemporaryDirectory() as tmpdir:
        #tmpdir = "../tmp/"
        # save similarity and cost files
        # similarity file is only required for fuzzy clustering
        with open(tmpdir + "/sim.tsv", "w") as outfile:
            for i, (g1, col) in enumerate(similarities.iteritems()):
                for j, (g2, value) in enumerate(col.iteritems()):
                    outfile.write(g1 + "\t" + g2 + "\t" + str(value) + "\n")

        cost = similarities.copy()
        cost.values[cost.values < threshold] = threshold - 1
        cost = cost - threshold
        with open(tmpdir + "/cost.tsv", "w") as outfile:
            outfile.write(str(cost.shape[0]) + "\n")
            outfile.write("\n".join(cost.index) + "\n")
            for i, (j, row) in zip(range(cost.shape[0], 1, -1), cost.iterrows()):
                outfile.write("\t".join(row.astype(str)[-i+1:]) + "\n")

        if cutoff == -1:
            fuzzytext = ""
            resultsfile = "results.tsv"
        else:
            fuzzytext = " -fuzzy " + str(cutoff)
            resultsfile = "results.tsv_fuzzy"
        # run the transitivity clustering tool
        command = "java -jar " + os.environ["PERSOFTWARELOCATION"] + "/TransClust.jar -i {tmpdir}/cost.tsv -o {tmpdir}/results.tsv -verbose -sim {tmpdir}/sim.tsv {fuzzytext}".format(**locals())

        sp.call(command, shell=True)

        results = pd.read_csv(tmpdir + "/" + resultsfile, sep="\t", squeeze=True, index_col=0)

    modules = [[] for i in range(results.max())]
    for g, moduleid in results.iteritems():
        if moduleid > 0:
            modules[moduleid-1].append(g)

    return modules

def mcl(E, simdist_function="pearson_correlation", inflation=10, threshold=None, **kwargs):
    similarities = simdist(E, simdist_function, **kwargs)

    if threshold is not None:
        similarities.values[similarities.values < threshold] = np.nan

    netinput = """
---8<------8<------8<------8<------8<---
""" + "\n".join(["\t".join([str(g) for g in edge]) + "\t" + str(similarity) for edge, similarity in list(similarities.stack().iteritems()) if edge[0] != edge[1] and similarity != np.nan]) + """
--->8------>8------>8------>8------>8---
    """

    p = sp.Popen(['mcl - --abc -I ' + str(inflation) + ' -o -'], stdout=sp.PIPE, stdin=sp.PIPE, stderr=sp.PIPE, shell=True)

    stdout_data = p.communicate(input=bytes(netinput))

    output = stdout_data[0].decode("utf-8")

    modules = []
    for line in output.splitlines():
        modules.append(line.split("\t"))

    return modules

## Decomposition
def ica_fdr(E, k=200, qvalcutoff=1e-3, seed=None, **kwargs):
    source = _ica_fastica(E, k, seed)
    modules = _ica_fdrtool(E, source, qvalcutoff)

    return modules

def ica_fdr_signed(E, k=200, qvalcutoff=1e-3, seed=None, **kwargs):
    source = _ica_fastica(E, k, seed)
    modules = _ica_fdrtool_signed(E, source, qvalcutoff)

    return modules

def ica_zscore(E, k=200, stdcutoff=3, **kwargs):
    source = _ica_fastica(E, k)
    modules = _ica_zscore(E, source, stdcutoff)

    return modules

def ica_percentage(E, k=200, perccutoff=0.075, **kwargs):
    source = _ica_fastica(E, k)
    modules = _ica_perccutoff(E, source, perccutoff)

    return modules

def ica_max(E, k=200, seed=None, **kwargs):
    source = _ica_fastica(E, k, seed)
    modules = convert_labels2modules(source.argmax(1), E.columns)

    return modules

def ipca(E, k=200, qvalcutoff=1e-3, **kwargs):
    source = _ipca(E, k)
    modules = _ica_fdrtool(E, source, qvalcutoff)

    return modules

def pca(E, k=200, qvalcutoff=1e-3, **kwargs):
    source = _pca(E, k)
    modules = _ica_fdrtool(E, source, qvalcutoff)

    return modules

def nmf_max(E, k=50, **kwargs):
    modules = []

    source = _nmf(E, k)
    # for column in fit.fit.connectivity():
    #     module = Module(E.columns[(column[0] != 0).tolist()[0]])
    #     if len(module) > 0:
    #         modules.append(module)
    modules = convert_labels2modules(source.argmax(0), E.columns)
    return modules

def nmf_tail(E, k=50, tailcutoff=0.05, **kwargs):
    source = _nmf(E, k)
    modules = _ica_tail(E, source, tailcutoff)

    return modules

def _nmf(E, k):
    if E.min().min() < 0:
        V = E.as_matrix() - E.min().min()
    else:
        V = E.as_matrix()

    nmf = nimfa.Nmf(V, rank=int(k), seed="random_vcol", max_iter=20000, update='euclidean')
    fit = nmf()

    print(nmf.rss())

    return fit.fit.H.A.T

def _ipca(E, k):
    ro.packages.importr("mixOmics")

    ipca = ro.r["ipca"]
    rresults = ipca(E, ncomp=k)

    source = np.array(rresults.rx2("loadings"))

    return source

def _ica_fastica(E, k, seed=None):

    ica = sklearn.decomposition.FastICA(n_components=k, random_state=seed)
    source = ica.fit_transform(standardize(E).T)

    return source

def _ica_zscore(E, source, stdcutoff):
    modules = []
    for source_row in source.T:
        genes = E.columns[source_row < -source_row.std() * stdcutoff].tolist() + E.columns[source_row > +source_row.std() * stdcutoff].tolist()

        modules.append(Module(genes))
    return modules

def _ica_fdrtool(E, source, qvalcutoff):
    importr("fdrtool")
    rfdrtool = ro.r["fdrtool"]

    modules = []

    print("qvalcutoff: " + str(qvalcutoff))

    for source_row in source.T:
        rresults = rfdrtool(ro.FloatVector(source_row), plot=False, cutoff_method="fndr", verbose=False)
        qvals = np.array(rresults.rx2("qval"))

        genes = E.columns[qvals < qvalcutoff]

        modules.append(Module(genes))
    return modules

def _ica_fdrtool_signed(E, source, qvalcutoff):
    importr("fdrtool")
    rfdrtool = ro.r["fdrtool"]

    modules = []

    print("qvalcutoff: " + str(qvalcutoff))

    for source_row in source.T:
        rresults = rfdrtool(ro.FloatVector(source_row), plot=False, cutoff_method="fndr", verbose=False)
        qvals = np.array(rresults.rx2("qval"))

        genes = E.columns[(qvals < qvalcutoff) & (source_row > source_row.mean())]

        modules.append(Module(genes))

        genes = E.columns[(qvals < qvalcutoff) & (source_row < source_row.mean())]

        modules.append(Module(genes))
    return modules

def _ica_tail(E, source, tailcutoff):
    cutoff = np.percentile(source.flatten(), (1-tailcutoff)*100)
    modules = []
    for source_row in source.T:
        genes = E.columns[source_row > cutoff].tolist()

        if len(genes) > 0:
            modules.append(Module(genes))
    return modules


def _ica_perccutoff(E, source, perccutoff):
    modules = []

    for source_row in source.T:
        sortedgenes = E.columns[source_row.argsort()]
        genes = sortedgenes[:int(round(len(E.columns) * perccutoff))]
        modules.append(Module(genes))

        genes = sortedgenes[-int(round(len(E.columns) * perccutoff)):]
        modules.append(Module(genes))
    return modules

def _pca(E, k):
    pca = sklearn.decomposition.PCA(n_components=int(k))

    source = pca.fit_transform(standardize(E))

    return pca.components_.T

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