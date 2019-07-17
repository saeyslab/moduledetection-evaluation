from __future__ import division

import numpy as np
import pandas as pd

# for speed, the module comparison functions are implemented in Cython
#import pyximport; pyximport.install()
import ebcubed
import jaccard

import json
from modulecontainers import Modules

from util import JSONExtendedEncoder

import os

from collections import OrderedDict
from scipy.stats import fisher_exact
from statsmodels.sandbox.stats.multicomp import multipletests

from munkres import munkres

from clustervalidityindices import *

import sys

import shutil

import time

def harmonic_mean(X):
    X = np.array(X)
    if np.any(X <= 0):
        return 0
    else:
        return len(X)/(np.sum(1/np.array(X)))

class ModulesComparison():
    """
    Compares two sets of modules using several scores
    """
    def __init__(self, modulesA, modulesB, G):
        self.modulesA = modulesA
        self.modulesB = modulesB
        self.G = G

        self.membershipsA = self.modulesA.cal_membership(self.G).astype(np.uint8)
        self.membershipsB = self.modulesB.cal_membership(self.G).astype(np.uint8)

        if len(self.modulesB) > 0 and len(self.modulesA) > 0:
            self.jaccards = np.nan_to_num(jaccard.cal_similaritymatrix_jaccard(self.membershipsA.T.values, self.membershipsB.T.values))
        else:
            self.jaccards = np.zeros((1,1))

    def score(self, baselines, scorenames = ["rr", "rp"]):
        """
        Scores two sets of modules

        """
        scores = {}

        # recovery and relevance
        if "rr" in scorenames:
            if (self.membershipsA.shape[1] == 0) or (self.membershipsB.shape[1] == 0):
                scores["recoveries"] = scores["relevances"] = np.zeros(1)
            else:
                scores["recoveries"] = self.jaccards.max(1)
                scores["relevances"] = self.jaccards.max(0)
            scores["recovery"] = scores["recoveries"].mean()
            scores["relevance"] = scores["relevances"].mean()
            scores["F1rr"] = harmonic_mean([scores["recovery"], scores["relevance"]])

        # recall and precision
        if "rp" in scorenames:
            if (self.membershipsA.shape[1] == 0) or (self.membershipsB.shape[1] == 0):
                scores["recalls"] = scores["precisions"] = np.zeros(1)
            else:
                scores["recalls"], scores["precisions"] = ebcubed.cal_ebcubed(self.membershipsA.values, self.membershipsB.values, self.jaccards.T.astype(np.float64))
            scores["recall"] = scores["recalls"].mean()
            scores["precision"] = scores["precisions"].mean()
            scores["F1rp"] = harmonic_mean([scores["recall"], scores["precision"]])

        # consensus score, uses the python munkres package
        if "consensus" in scorenames:
            if (self.membershipsA.shape[1] == 0) or (self.membershipsB.shape[1] == 0):
                scores["consensus"] = 0
            else:
                cost_matrix = np.array(1 - self.jaccards, dtype=np.double).copy()
                indexes =munkres(cost_matrix)
                consensus = (1-cost_matrix[indexes]).sum() / max(self.jaccards.shape)

        if ("rr" in scorenames) and ("rp" in scorenames):
            scores["F1rprr"] = harmonic_mean([scores["recall"], scores["precision"], scores["recovery"], scores["relevance"]])
        

        # compare with baseline
        if baselines is not None:
            for baseline_name, baseline in baselines.items():
                if "rr" in scorenames:
                    scores["F1rr_" + baseline_name] = harmonic_mean([(scores[scorename]/baseline[scorename]) for scorename in ["recovery", "relevance"]])
                if "rp" in scorenames:
                    scores["F1rp_" + baseline_name] = harmonic_mean([(scores[scorename]/baseline[scorename]) for scorename in ["recall", "precision"]])
                if ("rr" in scorenames) and ("rp" in scorenames):
                    scores["F1rprr_" + baseline_name] = harmonic_mean([(scores[scorename]/baseline[scorename]) for scorename in ["recovery", "relevance", "recall", "precision"]])
                if "consensus" in scorenames:
                    scores["consensus" + baseline_name] = harmonic_mean([(scores[scorename]/baseline[scorename]) for scorename in ["consensus"]])

        # alternative scores (for non-overlapping and exhaustive clustering)
        if "fmeasure_wiwie" in scorenames:
            scores["fmeasure_wiwie"] = fmeasure_wiwie(self.modulesA, self.modulesB)
        if "fmeasure_flowcap" in scorenames:
            scores["fmeasure_flowcap"] = fmeasure_flowcap(self.modulesA, self.modulesB)
        if "vmeasure_wiwie" in scorenames:
            scores["vmeasure_wiwie"] = vmeasure_wiwie(self.modulesA, self.modulesB)

        return scores

import multiprocessing as mp
class ModevalKnownmodules:
    def __init__(self, settings):
        self.settings = settings

    def run(self, pool):
        jobs = []
        manager = mp.Manager()
        scores = manager.dict()

        params_pool = []

        for setting in self.settings:
            params_pool.append((setting, scores))

        self.params_pool = params_pool

        pool.starmap(modevalworker, params_pool)

        scores = [scores_line for settingscores in list(scores.values()) for scores_line in settingscores]
        self.scores = pd.DataFrame(scores)
        self.scores = self.scores[[column for column in self.scores if column not in ["recoveries", "relevances", "recalls", "precisions"]]]
        self.scores_full = scores

    def save(self, name, full=True):
        self.scores.to_csv("../results/modeval_knownmodules/" + name + ".tsv", sep="\t")
        if full:
            json.dump(self.scores_full, open("../results/modeval_knownmodules/" + name + ".json", "w"), cls=JSONExtendedEncoder)

    def load(self, name, full=False):
        self.scores = pd.read_table("../results/modeval_knownmodules/" + name + ".tsv", index_col=0)
        if full:
            self.scores_full = json.load(open("../results/modeval_knownmodules/" + name + ".json"))

def modevalworker(setting, scores):
    baseline_names = ["permuted", "sticky", "scalefree"]
    baselines = {baseline_name:pd.read_table("../results/modeval_knownmodules/baselines_" + baseline_name + ".tsv", index_col=[0, 1,2]) for baseline_name in baseline_names}

    modules = Modules(json.load(open("../" + setting["output_folder"] + "modules.json")))

    runinfo = json.load(open("../" + setting["output_folder"] + "runinfo.json"))

    dataset = json.load(open("../" + setting["dataset_location"]))

    settingscores = []

    for regnet_name in dataset["knownmodules"].keys():
        for knownmodules_name in dataset["knownmodules"][regnet_name].keys():
            baselinesoi = {
                baseline_name:baseline.ix[(dataset["baselinename"], regnet_name, knownmodules_name)].to_dict()
                for baseline_name, baseline in baselines.items()
            }

            knownmodules_location = dataset["knownmodules"][regnet_name][knownmodules_name]
            knownmodules = Modules(json.load(open("../" + knownmodules_location)))

            settingscores_goldstandard = modevalscorer(modules, knownmodules, baselinesoi)

            settingscores_goldstandard["settingid"] = setting["settingid"]

            settingscores_goldstandard["knownmodules_name"] = knownmodules_name
            settingscores_goldstandard["regnet_name"] = regnet_name
            settingscores_goldstandard["goldstandard"] = regnet_name + "#" + knownmodules_name

            settingscores_goldstandard["runningtime"] = runinfo["runningtime"]

            settingscores.append(settingscores_goldstandard)

    scores[setting["settingid"]] = settingscores

def modevalscorer(modules, knownmodules, baselines=None):
    allgenes = sorted(list({g for module in knownmodules for g in module}))
    filteredmodules = modules.filter_retaingenes(allgenes).filter_size(5)
    comp = ModulesComparison(filteredmodules, knownmodules, allgenes)
    settingscores = comp.score(baselines)

    return settingscores

class ModevalCoverage:
    def __init__(self, settings):
        self.settings = settings

    def run(self, pool):
        jobs = []
        manager = mp.Manager()
        scores = manager.dict()

        params_pool = []

        print("Evaluating a total of " + str(len(self.settings)) + " settings.")

        for setting in self.settings:
            params_pool.append((setting, scores))

        pool.starmap(modeval_coverage_worker, params_pool)

        scores = [scores_line for settingscores in list(scores.values()) for scores_line in settingscores]
        self.scores = pd.DataFrame(scores)
        self.scores_full = scores

        manager.shutdown()

    def save(self, name, full=True):
        self.scores.to_csv("../results/modeval_coverage/" + name + ".tsv", sep="\t")
        if full:
            json.dump(self.scores_full, open("../results/modeval_coverage/" + name + ".json", "w"), cls=JSONExtendedEncoder)

    def load(self, name, full=False):
        self.scores = pd.read_table("../results/modeval_coverage/" + name + ".tsv", index_col=0)
        if full:
            self.scores_full = json.load(open("../results/modeval_coverage/" + name + ".json"))

class Modeval:
    def __init__(self, settings):
        self.settings = settings

    def save(self, name, full=True):
        self.scores.to_csv("../results/{scoring_folder}/{settings_name}.tsv".format(scoring_folder=scoring_folder, name = name), sep="\t")
        if full:
            json.dump(self.scores_full, open("../results/{scoring_folder}/{settings_name}.json".format(scoring_folder=scoring_folder, name = name), "w"), cls=JSONExtendedEncoder)

    def load(self, name, full=False):
        self.scores = pd.read_table("../results/{scoring_folder}/{settings_name}.tsv".format(scoring_folder=scoring_folder, name = name), index_col=0)
        if full:
            self.scores_full = json.load(open("../results/{scoring_folder}/{settings_name}.json".format(scoring_folder=scoring_folder, name = name)))

def modeval_coverage_worker(setting, scores, verbose=False):
    dataset = json.load(open("../" + setting["dataset_location"]))

    baseline_names = ["permuted", "sticky", "scalefree"]
    baselines = {baseline_name:pd.read_table("../results/modeval_coverage/baselines_" + baseline_name + ".tsv", index_col=[0, 1]) for baseline_name in baseline_names}
    baselines = {baseline_name:baseline.ix[dataset["baselinename"]] for baseline_name, baseline in baselines.items()}

    runinfo = json.load(open("../" + setting["output_folder"] + "runinfo.json"))
    modules = Modules(json.load(open("../" + setting["output_folder"] + "modules.json")))

    if verbose: print("▶ " + str(setting["settingid"]))

    subscores = []
    for bound_name, bound_location in dataset["binding"].items():
        if bound_location.endswith(".pkl"):
            bound = pd.read_pickle("../" + bound_location)
        else:
            bound = pd.read_table("../" + bound_location, index_col=0, header=[0,1])

        subscores.append(modbindevalscorer(modules, bound))

    # calculate the average over different regulatory circuit cutoffs
    settingscores = pd.DataFrame(subscores).mean().to_dict()
    for baseline_name, baseline in baselines.items():
        settingscores["aucodds_" + baseline_name] = settingscores["aucodds"]/baseline["aucodds"].mean()

    settingscores["settingid"] = setting["settingid"]
    settingscores["goldstandard"] = "regcircuit"
    settingscores["runningtime"] = runinfo["runningtime"]

    scores[setting["settingid"]] = [settingscores]

    if verbose: print("◼ " + str(setting["settingid"]))

def modbindevalscorer(modules, binding):
    modules = modules.filter_size(5)
    if len(modules) == 0:
        aucodds=0

        odds = pd.DataFrame()
        pvals = pd.DataFrame()
        qvals = pd.DataFrame()
    else:
        modmem = modules.cal_membership(G=binding.index)
        binmem = binding

        modsizes = modmem.sum()
        binsizes = binmem.sum()

        tps = modmem.T.dot(binmem.astype(np.int))
        fps = binsizes - tps
        fns = (modsizes - tps.T).T
        tns = binmem.shape[0] - tps - fps - fns

        odds = ((tps * tns)/(fps*fns))

        values = np.array([odds.values.flatten(),  tps.values.flatten(), fps.values.flatten(), fns.values.flatten(), tns.values.flatten()])

        pvals = np.apply_along_axis(filterfisher, 0, values)
        qvals = []
        for pvalrow in pvals.reshape(tps.shape):
            _,qvalrow,_,_ = np.array(multipletests(pvalrow))
            qvals.append(qvalrow)
        qvals = pd.DataFrame(qvals, index=tps.index, columns=tps.columns)
        pvals = pd.DataFrame(pvals.reshape(tps.shape), index=tps.index, columns=tps.columns)
        if binding.columns.nlevels > 1:
            pvals = pvals.T.groupby(level=0).min()
            qvals = qvals.T.groupby(level=0).min() # group by regulator
            odds = odds.T.groupby(level=0).max() # group by regulator
        else:
            pvals = pvals.T
            qvals = qvals.T
            odds = odds.T

        ## auc odds

        odds_filtered = odds.copy()
        odds_filtered.values[(qvals > 0.05).values.astype(np.bool)] = 0
        odds_max = odds_filtered.max(1)

        if len(odds_max) == 0:
            aucodds=0
        else:
            cutoffs = np.linspace(0, 3, 100)

            stillenriched = [(np.log10(odds_max) >= cutoff).sum()/len(odds_max) for cutoff in cutoffs]
            aucodds = np.trapz(stillenriched, cutoffs) / (cutoffs[-1] - cutoffs[0])

    scores = {"aucodds":aucodds}

    return scores

class ModevalFunctional:
    scoring_folder = "../results/modeval_function/"
    def __init__(self, settings):
        self.settings = settings

    def run(self, pool, gset_names = None):
        jobs = []
        manager = mp.Manager()
        scores = manager.dict()

        params_pool = []

        print("Evaluating a total of " + str(len(self.settings)) + " settings.")

        for setting in self.settings:
            params_pool.append((setting, scores))

        pool.starmap(modenrichevalworker, params_pool)

        scores = [scores_line for settingscores in list(scores.values()) for scores_line in settingscores]
        self.scores = pd.DataFrame(scores)
        self.scores_full = scores

        manager.shutdown()

    def save(self, settings_name):
        self.scores.to_csv(self.scoring_folder + settings_name + ".tsv", sep="\t")
        if hasattr(self, 'scores_full'):
            json.dump(self.scores_full, open(self.scoring_folder + settings_name + ".json", "w"), cls=JSONExtendedEncoder)

    def load(self, settings_name):
        self.scores = pd.read_table(self.scoring_folder + settings_name + ".tsv", index_col=0)

def modenrichevalworker(setting, scores):
    dataset = json.load(open("../" + setting["dataset_location"]))

    modules = Modules(json.load(open("../" + setting["output_folder"] + "modules.json")))

    settingscores = []
    for gsets_name in dataset["gsets"].keys():
        gsets_location = dataset["gsets"][gsets_name]
        membership = pd.read_pickle("../" + gsets_location)
        connectivity = pd.read_pickle("../" + gsets_location[:-4] + "_connectivity.pkl")

        settingscores_gsets = modenrichevalscorer(modules, membership, connectivity)

        settingscores.extend([{"settingid":setting["settingid"], "scorename":scorename + "#" + gsets_name, "score":score} for scorename, score in settingscores_gsets.items()])

    scores[setting["settingid"]] = settingscores

def cal_bhi(modules, connectivity):
    bhi = 0
    for module in modules:
        if len(module) > 1:
            bhi += 1/(len(module) * (len(module) - 1)) * connectivity.ix[module, module].sum().sum()
    bhi = 1/len(modules) * bhi

    return bhi

import fisher
from scipy.stats import chi2_contingency

def test_enrichment(modules, membership):
    if len(modules) == 0:
        odds = pd.DataFrame()
        pvals = pd.DataFrame()
        qvals = pd.DataFrame()
    else:
        modmem = modules.cal_membership(G=membership.index)

        modsizes = modmem.sum()
        binsizes = membership.sum()

        tps = modmem.T.dot(membership.astype(np.int))
        fps = binsizes - tps
        fns = (modsizes - tps.T).T
        tns = membership.shape[0] - tps - fps - fns

        odds = ((tps * tns)/(fps*fns))

        values = np.array([odds.values.flatten(),  tps.values.flatten(), fps.values.flatten(), fns.values.flatten(), tns.values.flatten()])

        pvals = np.apply_along_axis(filterfisher, 0, values)
        qvals = []
        for pvalrow in pvals.reshape(tps.shape):
            _,qvalrow,_,_ = np.array(multipletests(pvalrow))
            qvals.append(qvalrow)
        qvals = pd.DataFrame(qvals, index=tps.index, columns=tps.columns)
        pvals = pd.DataFrame(pvals.reshape(tps.shape), index=tps.index, columns=tps.columns)
    return pvals, qvals, odds

def filterfisher(x):
    return fisher.pvalue(x[1], x[2], x[3], x[4]).right_tail

def cal_aucodds(odds, cutoffs=np.linspace(0, 3, 100)):
    maxodds = odds.max(1)
    if len(maxodds) == 0:
        return 0

    stillenriched = [(np.log10(maxodds) >= cutoff).sum()/len(maxodds) for cutoff in cutoffs]
    return np.trapz(stillenriched, cutoffs) / (cutoffs[-1] - cutoffs[0])

def cal_faucodds(odds, cutoffs=np.linspace(0, 3, 100)):
    aucodds1 = cal_aucodds(odds, cutoffs)
    aucodds2 = cal_aucodds(odds.T, cutoffs)

    return 2/(1/aucodds1 + 1/aucodds2)

def modenrichevalscorer(modules, membership, connectivity):
    modules = modules.filter_size(5)

    pvals, qvals, odds = test_enrichment(modules, membership)
    filteredodds = odds.copy()
    filteredodds.values[(qvals > 0.1).values.astype(np.bool)] = 0

    scores = {
        "bhi":cal_bhi(modules, connectivity),
        "faucodds":cal_faucodds(filteredodds)
    }

    return scores
