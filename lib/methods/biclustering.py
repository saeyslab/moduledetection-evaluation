import numpy as np
import pandas as pd

import rpy2.robjects as ro
import rpy2.robjects.numpy2ri
rpy2.robjects.numpy2ri.activate()
import rpy2.robjects.pandas2ri
rpy2.robjects.pandas2ri.activate()

from rpy2.robjects.packages import importr

from modulecontainers import Bicluster, Modules, Module

from sklearn.cluster.bicluster import SpectralBiclustering

from util import TemporaryDirectory

import os
import subprocess as sp

from collections import defaultdict

def standardize(X):
    return (X - X.mean(0))/(X.std(0))

def isa(E, thr_col=2, thr_row=2, no_seeds=10000, **kwargs):
    importr("isa2")
    ro.globalenv["E"] = E

    print(thr_col, thr_row, no_seeds)

    risa =  ro.r["isa"]

    rresults = risa(
        standardize(E).as_matrix(), 
        # Workaround to get dots in the parameter names
        **{
            "thr.col":thr_col,
            "thr.row":thr_row,
            "no.seeds":no_seeds
        }
    )

    rows = np.array(rresults.rx2("rows")).T
    columns = np.array(rresults.rx2("columns")).T

    bics = convert_biclustermatrix2biclusters(rows, columns, E)

    return bics

def spectral_biclust(E, ngenes=3, nconditions=1,  spectral_method="bistochastic", n=6, n_best_ratio=0.5, **kwargs):
    n_best = max([int(n*n_best_ratio), 1])

    spectral = SpectralBiclustering(n_clusters=(nconditions,ngenes), method=spectral_method, n_components=n, n_best=n_best)

    spectral.fit(standardize(E))

    bics = []
    for columns, rows in zip(spectral.columns_, spectral.rows_):
        genes = E.columns[columns]
        conditions = E.index[rows]

        bics.append(Bicluster(genes, conditions))

    return bics


def qubic(E, q=0.06, rank=1, tolerance=0.95, filter_proportion=1, k=1000, **kwargs):
    importr("rqubic")
    importr("Biobase")
    ro.globalenv["E"] = E

    rresults = ro.r("""
    exprs =  new("ExpressionSet", exprs=t(scale(E)))
    sampleNames(exprs) = rownames(E)   

    disc = quantileDiscretize(exprs, q={q}, rank={rank})
    seeds = generateSeeds(disc)

    results =  quBicluster(
        seeds, 
        disc, 
        tolerance={tolerance}, 
        report.no={k}, 
        filter.proportion={filter_proportion}
    )

    results
    """.format(**locals()))

    rows = np.array(rresults.do_slot("NumberxCol"))
    columns = np.array(rresults.do_slot("RowxNumber")).T

    bics = convert_biclustermatrix2biclusters(rows, columns, E)

    return(bics)

def blockcluster(E, ngenes=10, nconditions=10, **kwargs):
    importr("blockcluster")

    print(ngenes, nconditions)

    rresults = ro.r["coclusterContinuous"](ro.r["as.matrix"](standardize(E)), nbcocluster=ro.IntVector([nconditions,ngenes]))

    modules = Modules(convert_labels2modules(rresults.slots["colclass"], E.columns))
    moduleconditions = convert_labels2modules(rresults.slots["rowclass"], E.index)

    return(modules)


def biclust(E, biclust_method, biclust_kwargs={}):
    """
    Wrapper for the biclustering methods within the R `biclust` package
    """

    importr("biclust")

    rbiclust = ro.r["biclust"]
    rbiclust_method = ro.r[biclust_method]

    rE = standardize(E).as_matrix()
    rE = rE.T

    if biclust_method == "BCXmotifs":
        rE = ro.r["discretize"](rE)

    rresults = rbiclust(rE, method=rbiclust_method, **{key.replace("_", "."):value for key,value in biclust_kwargs.items()})

    rows = np.array(rresults.do_slot("NumberxCol"))
    columns = np.array(rresults.do_slot("RowxNumber")).T

    bics = convert_biclustermatrix2biclusters(rows, columns, E)

    return bics

def plaid(E, fit_model="y ~ m + a + b", row_release=0.7, col_release=0.7, shuffle=3, back_fit=0, max_layers=20, iter_startup=5, iter_layer=10, **kwargs):
    fit_model = ro.r(fit_model)
    locals_ = locals()
    return biclust(E, "BCPlaid", {param_name:locals_[param_name] for param_name in ["fit_model", "row_release", "col_release", "shuffle", "back_fit", "max_layers", "iter_startup", "iter_layer"]})

def chengchurch(E, delta=1.5, alpha=1.5, number=100, **kwargs):
    locals_ = locals()
    return biclust(E, "BCCC", {param_name:locals_[param_name] for param_name in ["delta", "alpha", "number"]})

def xmotifs(E, ns=10, nd=10, sd=5, alpha=0.05, number=100, **kwargs):
    locals_ = locals()
    return biclust(E, "BCXmotifs", {param_name:locals_[param_name] for param_name in ["ns", "nd", "sd", "alpha", "number"]})

def spectral(E, numberOfEigenvalues=3, normalization="log", minr=2, minc=2, withinVar=1, **kwargs):
    locals_ = locals()
    return biclust(E, "BCSpectral", {param_name:locals_[param_name] for param_name in ["minr", "minc", "numberOfEigenvalues", "withinVar"]})

def msbe(E, alpha=0.4, beta=0.5, gamma=1.2, refgene="random 500", refcond="random 20", **kwargs):
    with TemporaryDirectory() as tmpdir:
        standardize(E).to_csv(tmpdir + "/E.csv", sep="\t")

        binary = "sh " +os.environ["PERSOFTWARELOCATION"] + "/MSBE_linux_1.0.5/additiveBi"
        command =  "{binary} {tmpdir}/E.csv {alpha} {beta} {gamma} {refgene} {refcond} {tmpdir}/results.txt".format(**locals())
        sp.call(command, shell=True)

        bics = []
        with open(tmpdir + "/results.txt" , "r") as infile:
            lines = infile.readlines()
            for _, line1, line2 in zip(lines[::3], lines[1::3], lines[2::3]):
                if len(line1) > 0 and len(line2) > 0:
                    print(line1, line2)
                    cids = [int(cid)-1 for cid in line1.strip().split(" ")]
                    gids = [int(gid)-1 for gid in line2.strip().split(" ")]

                bics.append(Bicluster(E.columns[gids].tolist(), E.index[cids].tolist()))

    return bics

def opsm(E, l=2, **kwargs):
    with TemporaryDirectory() as tmpdir:
        pd.DataFrame(standardize(E)).T.to_csv(tmpdir + "E.csv", index=0, header=0, sep=" ")
        output_location = os.path.abspath(tmpdir + "/output.txt")

        binary = "java -XX:ParallelGCThreads=1 -Xmx1G -jar " + os.environ["PERSOFTWARELOCATION"] + "/OPSM/OPSM.jar"
        command = "{binary} {E_location} {nG} {nC} {output_location} {l}".format(
            binary=binary, 
            E_location = os.path.abspath(tmpdir + "E.csv"),
            nG = str(len(E.columns)),
            nC = str(len(E.index)),
            output_location = output_location,
            l = str(l)
        )
        print(command)
        sp.call(command, shell=True)

        bics = []
        with open(os.path.abspath(output_location) , "r") as infile:
            lines = infile.readlines()
            for line1, line2, _ in zip(lines[::3], lines[1::3], lines[2::3]):
                if len(line1) > 0 and len(line2) > 0:
                    gids = [int(gid) for gid in line1.strip().split(" ")]
                    cids = [int(cid) for cid in line2.strip().split(" ")]

                bics.append(Bicluster(E.columns[gids].tolist(), E.index[cids].tolist()))

    return bics

def biforce(E, m="o", t=0, **kwargs):
    with TemporaryDirectory() as tmpdir:
        E_location = tmpdir + "/E.csv"
        output_location = tmpdir + "/output.txt"

        standardize(E).T.to_csv(tmpdir + "/E.csv", index=False, header=False, sep="\t")

        binary = "java -XX:ParallelGCThreads=1 -Xmx12G -jar mbiforce.jar"
        command = "{binary} -i={i} -o={o} -m={m} -h=false -t={t}".format(
            binary=binary,
            i=os.path.abspath(E_location),
            o=os.path.abspath(output_location),
            m=m,
            t=str(t)
        )

        original_wd = os.getcwd()
        try:
            os.chdir(os.environ["PERSOFTWARELOCATION"] + "/BiForceV2/") # change working directory because biclue only looks for the parameter.ini file in the current working directory (...)
            
            sp.call(command, shell=True)
        except BaseException as e:
            raise e
        finally:
            os.chdir(original_wd)

        bics = []
        with open(output_location) as infile:
            for line in infile.readlines()[1:-1]:
                line = line.strip().split(",")
                genes = []
                conditions = []
                for rowcol in line:
                    if rowcol[0] == "R":
                        genes.append(E.columns[int(rowcol[1:])-1])
                    elif rowcol[0] == "C":
                        conditions.append(E.index[int(rowcol[1:])-1])
                bics.append(Bicluster(genes, conditions))

    return bics


def fabia(E, n=13, alpha=0.01, cyc=500, spl=0., spz=0.5, non_negative=0, random=1, center=2, norm=1, scale=0, lap=1, nL=0, lL=0, bL=0, thresZ=0.5, thresL=None, **kwargs):
    importr("fabia")

    rfabia = ro.r["fabia"]

    if thresL is None or thresL == "None":
        thresL = ro.NULL

    rresults = rfabia(
        standardize(E).as_matrix().T, 
        n,
        alpha,
        cyc,
        spl, 
        spz, 
        non_negative,
        random,
        center,
        norm,
        scale,
        lap,
        nL,
        lL,
        bL
    )

    rresults_extracted =  ro.r["extractBic"](rresults, thresZ, thresL).rx2("bic")
    bics = []
    for i in range(1,rresults_extracted.nrow):
        if np.min(rresults_extracted.rx2(i,"binp")) > 0: # number of rows and columns > 0
            gids = [int(g[len("gene"):])-1 for g in rresults_extracted.rx2(i,"bixn")]
            cids = [int(g[len("sample"):])-1 for g in rresults_extracted.rx2(i,"biypn")]

            bics.append(Bicluster(E.columns[gids].tolist(), E.index[cids].tolist()))

    return bics

## utility functions
def convert_labels2modules(labels, G, ignore_label=None):
    modules = defaultdict(Module)
    for label, gene in zip(labels, G):
        if label != ignore_label:
            modules[label].add(gene)
    return list(modules.values())

def convert_biclustermatrix2biclusters(rows, columns, E):
    bics = []
    for column, row in zip(columns, rows):
        print(column)
        print(row)
        bic = Bicluster(E.columns[column != 0], E.index[row != 0])
        if bic.size() > 0:
            bics.append(bic)

    return bics