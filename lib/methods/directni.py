import numpy as np
import pandas as pd

from collections import defaultdict

import rpy2.robjects as ro
import rpy2.robjects.numpy2ri
rpy2.robjects.numpy2ri.activate()
import rpy2.robjects.pandas2ri
rpy2.robjects.pandas2ri.activate()

from rpy2.robjects.packages import importr

import os
import shutil

import tempfile

import subprocess as sp

from util import TemporaryDirectory

def standardize(X):
    return (X - X.mean())/(X.std())

def genie3(E, R, numcores=24, **kwargs):
    R = sorted(list(set(R) & set(E.columns)))

    print(numcores)

    with TemporaryDirectory() as tmpdir:
        E.to_csv(tmpdir + "/E.csv")
        with open(tmpdir + "/R.csv", "w") as outfile:
            outfile.write("\n".join(R))

        Rcommand = """
        source("{genie3_location}")

        E = read.delim("{tmpdir}/E.csv", header=T, row.names=1, sep=",")

        R = as.character(read.csv("{tmpdir}/R.csv", header=F)[[1]])

        results = run.genie3(E, tf.idx=R, mc.cores={numcores})

        write.table(results, "{tmpdir}/results.csv", sep="\\t")

        """.format(genie3_location=os.environ["PERSOFTWARELOCATION"] + "/GENIE3_Robrecht/mcgenie3.R", **locals())

        print(Rcommand)

        with open(tmpdir + "/rscript.R", "w") as script_outfile:
            script_outfile.write(Rcommand)
        sp.call("Rscript " + tmpdir + "/rscript.R", shell=True)

        wnet = pd.read_csv(tmpdir + "/results.csv", sep="\t")

        wnet.columns = E.columns
        wnet.index = R

        print(wnet)

        wnet = wnet.T # make sure target genes are in the rows, regulators are the columns

    return wnet

def clr(E, R, **kwargs):
    R = sorted(list(set(R) & set(E.columns)))

    with TemporaryDirectory() as tmpdir:
        E.to_csv(tmpdir + "/E.csv", sep="\t")

        matlab_command = """
        addpath '""" + os.environ["PERSOFTWARELOCATION"] + """/CLRv1.2.2/Code/';
        data = transpose(dlmread('""" + tmpdir + """/E.csv','\\t', 1, 1));

        A = clr(data);

        dlmwrite('""" + tmpdir + """/fullmatrix.csv', A, '\\t');
        exit;
        """

        matlab_command = matlab_command.replace("\n", "")

        command = "module load matlab;matlab -nodesktop -nosplash -nojvm -nodisplay -r \"" + matlab_command + "\""
        print(command)
        sp.call(command, shell=True)

        # postprocess context-likelihood ratio matrix, remove non-regulators

        wnet_preproc = pd.read_csv(tmpdir + "/fullmatrix.csv", sep="\t", names=E.columns, header=None, index_col=None)
        wnet_preproc.index = E.columns

        wnet = wnet_preproc.ix[E.columns, R]

    return wnet

def tigress(E, R, tigress_R= 500, tigress_L=5, tigress_alpha=0.2, **kwargs):
    R = sorted(list(set(R) & set(E.columns)))

    with TemporaryDirectory() as tmpdir:
        E = E[R + [g for g in E.columns if g not in R]] # tigress returns errors if the tfs are not located first...
        E.to_csv(tmpdir + "/E_expression_data.tsv", index=False, sep="\t")

        with open(tmpdir + "/E_transcription_factors.tsv", "w") as tfs_outfile:
            tfs_outfile.write("\n".join(R))

        # call matlab

        matlab_command = """
        addpath(genpath('""" + os.environ["PERSOFTWARELOCATION"] + """/TIGRESS-PKG-2.1/'));dataset = read_data('""" + tmpdir + """/', 'E');
        dataset.expdata = scale_data(dataset.expdata);
        freq=tigress(dataset, 'R', """ + str(tigress_R) + """, 'L',""" + str(tigress_L) + """, 'alpha', """ + str(tigress_alpha) + """);
        scores=score_edges(freq);
        dlmwrite('""" + tmpdir + """/scores.csv', scores, '\\t');
        exit;
        """
        matlab_command = matlab_command.replace("\n", "")
        print(matlab_command)

        command = "module load matlab/x86_64/R2013a;matlab -nodesktop -nosplash -nojvm -nodisplay -singleCompThread -r \"" + matlab_command + "\""
        command = "module load matlab/x86_64/R2013a;matlab -nodesktop -nosplash -nojvm -nodisplay -r \"" + matlab_command + "\""
        print(command)
        sp.call(command, shell=True)

        # postprocess frequency matrix

        wnet = pd.read_csv(tmpdir + "/scores.csv", sep="\t", names=E.columns, header=None, index_col=None).T
        wnet.columns = R

    return wnet

def correlation(E, R, **kwargs):
    R = sorted(list(set(R) & set(E.columns)))

    gene_cors = pd.DataFrame(np.abs(np.corrcoef(E.T)), index=E.columns, columns=E.columns)

    wnet = gene_cors[R]

    return wnet