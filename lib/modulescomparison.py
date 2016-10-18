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
            self.jaccards = np.nan_to_num(jaccard.cal_similaritymatrix_jaccard(self.membershipsA.T.as_matrix(), self.membershipsB.T.as_matrix()))
        else:
            self.jaccards = np.zeros((1,1))

        # overlaps = []
        # forwardoverlaps = []
        # reverseoverlaps = []
        # for moduleA in self.modulesA:
        #     for moduleB in self.modulesB:
        #         if moduleA == moduleB or len(moduleA) == 0 or len(moduleB) == 0:
        #             overlaps.append(0)
        #         else:
        #             overlaps.append(len(moduleA & moduleB))

        #         if len(moduleA) == 0:
        #             forwardoverlaps.append(0)
        #         else:
        #             forwardoverlaps.append(len(moduleA & moduleB)/len(moduleA))

        #         if len(moduleB) == 0:
        #             reverseoverlaps.append(0)
        #         else:
        #             reverseoverlaps.append(len(moduleA & moduleB)/len(moduleB))

        # self.overlaps = np.array(overlaps).reshape(len(self.modulesA), len(self.modulesB))
        # self.forwardoverlaps = np.array(forwardoverlaps).reshape(len(self.modulesA), len(self.modulesB))
        # self.reverseoverlaps = np.array(reverseoverlaps).reshape(len(self.modulesA), len(self.modulesB))  

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
                scores["recalls"], scores["precisions"] = ebcubed.cal_ebcubed(self.membershipsA.as_matrix(), self.membershipsB.as_matrix(), self.jaccards.T.astype(np.float64))
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
        baseline_names = ["permuted", "sticky", "scalefree"]
        baselines = {baseline_name:pd.read_table("../results/modeval_knownmodules/baselines_" + baseline_name + ".tsv", index_col=[0, 1,2]) for baseline_name in baseline_names}

        jobs = []
        manager = mp.Manager()
        scores = manager.dict()

        params_pool = []

        i = 0
        for setting in self.settings:
            modules = Modules(json.load(open("../" + setting["output_folder"] + "modules.json")))

            runinfo = json.load(open("../" + setting["output_folder"] + "runinfo.json"))

            method = json.load(open("../" + setting["method_location"]))
            dataset = json.load(open("../" + setting["dataset_location"]))

            for regnet_name in dataset["knownmodules"].keys():
                for knownmodules_name in dataset["knownmodules"][regnet_name].keys():
                    baselinesoi = {
                        baseline_name:baseline.ix[(dataset["baselinename"], regnet_name, knownmodules_name)].to_dict()
                        for baseline_name, baseline in baselines.items()
                    }

                    params_pool.append((i, modules, regnet_name, knownmodules_name, method, dataset, runinfo, baselinesoi, scores))

                    i+=1

        self.params_pool = params_pool

        pool.starmap(modevalworker, params_pool)

        self.scores = pd.DataFrame(list(scores.values()))
        self.scores = self.scores[[column for column in self.scores if column not in ["recoveries", "relevances", "recalls", "precisions"]]]
        self.scores_full = list(scores.values())

    def save(self, name, full=True):
        self.scores.to_csv("../results/modeval_knownmodules/" + name + ".tsv", sep="\t")
        if full:
            json.dump(self.scores_full, open("../results/modeval_knownmodules/" + name + ".json", "w"), cls=JSONExtendedEncoder)

    def load(self, name, full=False):
        self.scores = pd.read_table("../results/modeval_knownmodules/" + name + ".tsv", index_col=0)
        if full:
            self.scores_full = json.load(open("../results/modeval_knownmodules/" + name + ".json"))

def modevalworker(settingid, modules, regnet_name, knownmodules_name, method, dataset, runinfo, baselines, scores, verbose=False):
    if verbose:print("▶ " + str(settingid))
    sys.stdout.flush()

    knownmodules_location = dataset["knownmodules"][regnet_name][knownmodules_name]
    knownmodules = Modules(json.load(open("../" + knownmodules_location)))

    settingscores = modevalscorer(modules, knownmodules, baselines)

    settingscores.update(method["params"])
    settingscores.update(dataset["params"])
    settingscores["knownmodules_name"] = knownmodules_name
    settingscores["regnet_name"] = regnet_name
    settingscores["runningtime"] = runinfo["runningtime"]

    scores[settingid] = settingscores

    if verbose:print("◼ " + str(settingid))
    sys.stdout.flush()

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
        baseline_names = ["permuted", "sticky", "scalefree"]
        baselines = {baseline_name:pd.read_table("../results/modeval_coverage/baselines_" + baseline_name + ".tsv", index_col=[0, 1]) for baseline_name in baseline_names}

        jobs = []
        manager = mp.Manager()
        scores = manager.dict()

        params_pool = []

        i = 0

        self.modulesets = [Modules(json.load(open("../" + setting["output_folder"] + "modules.json"))) for setting in self.settings]

        print("Evaluating a total of " + str(len(self.settings)) + " settings.")

        for setting, modules in zip(self.settings, self.modulesets):
            runinfo = json.load(open("../" + setting["output_folder"] + "runinfo.json"))

            method = json.load(open("../" + setting["method_location"]))
            dataset = json.load(open("../" + setting["dataset_location"]))

            params_pool.append((i, modules, method, dataset, runinfo, {baseline_name:baseline.ix[dataset["datasetname"]] for baseline_name, baseline in baselines.items()}, scores))
            i+=1

        pool.starmap(modeval_coverage_worker, params_pool)

        self.scores = pd.DataFrame(list(scores.values()))
        self.scores_full = list(scores.values())

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
        self.scores.to_csv("../results/modeval_coverage/" + name + ".tsv", sep="\t")
        if full:
            json.dump(self.scores_full, open("../results/modeval_coverage/" + name + ".json", "w"), cls=JSONExtendedEncoder)

    def load(self, name, full=False):
        self.scores = pd.read_table("../results/modeval_coverage/" + name + ".tsv", index_col=0)
        if full:
            self.scores_full = json.load(open("../results/modeval_coverage/" + name + ".json"))

def modeval_coverage_worker(settingid, modules, method, dataset, runinfo, baselines, scores, verbose=False):
    if verbose: print("▶ " + str(settingid))

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

    settingscores.update(method["params"])
    settingscores.update(dataset["params"])
    settingscores["binding"] = "regcircuit"
    settingscores["runningtime"] = runinfo["runningtime"]

    scores[settingid] = settingscores

    if verbose: print("◼ " + str(settingid))

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

        values = np.array([odds.as_matrix().flatten(),  tps.as_matrix().flatten(), fps.as_matrix().flatten(), fns.as_matrix().flatten(), tns.as_matrix().flatten()])

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
        odds_filtered.values[(qvals > 0.05).as_matrix().astype(np.bool)] = 0
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

        i = 0

        self.modulesets = [Modules(json.load(open("../" + setting["output_folder"] + "modules.json"))) for setting in self.settings]

        print("Evaluating a total of " + str(len(self.settings)) + " settings.")

        if os.path.exists(self.scoring_folder + self.settings_name):
            shutil.rmtree(self.scoring_folder + self.settings_name)
        os.mkdir(self.scoring_folder + self.settings_name)

        if gset_names is None:
            list(dataset["gsets"].keys())

        for setting, modules in zip(self.settings, self.modulesets):
            runinfo = json.load(open("../" + setting["output_folder"] + "runinfo.json"))

            method = json.load(open("../" + setting["method_location"]))
            dataset = json.load(open("../" + setting["dataset_location"]))

            for gsets_name in gset_names:
                gsets_location = dataset["gsets"][gsets_name]
                params_pool.append((i, self.scoring_folder + self.settings_name + "/", modules, gsets_name, gsets_location, method, dataset, runinfo, None, scores))

                i+=1

        pool.starmap(modenrichevalworker, params_pool)

        try:
            self.load()
            oldscores = self.scores
        except:
            print("no previous found!")

        self.scores = pd.DataFrame(list(scores.values()))
        self.scores["gsetsource"] = self.scores["binding"]

        if "oldscores" in locals().keys():
            self.scores = pd.concat([self.scores, oldscores[[col for col in oldscores if col not in self.scores.columns]]], axis=1)
        self.scores_full = list(scores.values())

        manager.shutdown()

    def save(self, settings_name=None):
        self.scores.to_csv(self.scoring_folder + settings_name + ".tsv", sep="\t")
        if hasattr(self, 'scores_full'):
            json.dump(self.scores_full, open(self.scoring_folder + settings_name + ".json", "w"), cls=JSONExtendedEncoder)

    def load(self):
        self.scores = pd.read_table(self.scoring_folder + settings_name + ".tsv", index_col=0)

def modenrichevalworker(settingid, scoring_folder, modules, bound_name, bound_location, method, dataset, runinfo, baseline, scores):
    print("▶ " + str(settingid))
    if bound_location.endswith(".pkl"):
        bound = pd.read_pickle("../" + bound_location)
        connectivity = pd.read_pickle("../" + bound_location[:-4] + "_connectivity.pkl")
    else:
        bound = pd.read_table("../" + bound_location, index_col=0, header=[0,1])

    scores_real = modenrichevalscorer(modules, bound, connectivity, baseline, calculate_aucodds=False)

    #bound_random = bound.copy()
    #bound_random.index = np.random.permutation(bound_random.index)

    settingscores = scores_real

    settingscores.update(method["params"])
    settingscores.update(dataset["params"])
    settingscores["binding"] = bound_name
    settingscores["runningtime"] = runinfo["runningtime"]

    #pvals.to_pickle(scoring_folder + str(settingid) + "pvals.pkl")
    #odds.to_pickle(scoring_folder + str(settingid) + "odds.pkl")
    #qvals.to_pickle(scoring_folder + str(settingid) + "qvals.pkl")

    scores[settingid] = settingscores

    print("◼ " + str(settingid))

import fisher
from scipy.stats import chi2_contingency

def filterfisher(x):
    return fisher.pvalue(x[1], x[2], x[3], x[4]).right_tail

try:
    from matplotlib.pyplot import *
except:
    print("no matplotlib")

def cal_aucodds(odds):
    maxodds = odds.max(1)

    if len(maxodds) == 0:
        return 0

    cutoffs = np.linspace(0, 3, 100)

    #plot(cutoffs, [(np.log10(maxodds) >= cutoff).sum()/len(maxodds) for cutoff in cutoffs])
    #ylim(0,1)

    stillenriched = [(np.log10(maxodds) >= cutoff).sum()/len(maxodds) for cutoff in cutoffs]

    return np.trapz(stillenriched, cutoffs) / (cutoffs[-1] - cutoffs[0])

def modenrichevalscorer(modules, binding, connectivity, baseline=None, calculate_aucodds=True):
    modules = modules.filter_size(5)
    if len(modules) == 0:
        aucqval = 0
        aucodds=0
        aucodds1 = 0
        aucodds2 = 0

        odds = pd.DataFrame()
        pvals = pd.DataFrame()
        qvals = pd.DataFrame()

        bhi = 0
    else:
        ## BHI
        #binding_filtered = binding.ix[:,binding.sum() < len(binding.index)*0.2]
        #connectivity = np.dot(binding_filtered, binding_filtered.T)
        #np.fill_diagonal(connectivity, False)
        #connectivity = pd.DataFrame(connectivity, index=binding_filtered.index, columns=binding_filtered.index)

        start = time.time()

        bhi = 0
        for module in modules:
            if len(module) > 1:
                bhi += 1/(len(module) * (len(module) - 1)) * connectivity.ix[module, module].sum().sum()
        bhi = 1/len(modules) * bhi

        end = time.time()
        print(">>" + str(end - start))

        ## AUCODDS
        if calculate_aucodds:
            modmem = modules.cal_membership(G=binding.index)
            binmem = binding

            modsizes = modmem.sum()
            binsizes = binmem.sum()

            tps = modmem.T.dot(binmem.astype(np.int))
            fps = binsizes - tps
            fns = (modsizes - tps.T).T
            tns = binmem.shape[0] - tps - fps - fns

            odds = ((tps * tns)/(fps*fns))

            values = np.array([odds.as_matrix().flatten(),  tps.as_matrix().flatten(), fps.as_matrix().flatten(), fns.as_matrix().flatten(), tns.as_matrix().flatten()])

            pvals = np.apply_along_axis(filterfisher, 0, values)
            qvals = []
            for pvalrow in pvals.reshape(tps.shape):
                _,qvalrow,_,_ = np.array(multipletests(pvalrow))
                qvals.append(qvalrow)
            #qvals = pvals.reshape(odds.shape)
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

            ## auc qvals
            cutoffs = 10**(-np.arange(0, 10.0000001, 0.05))
            minqvals = qvals.min(1)
        
            #plot(np.log10(cutoffs), [(minqvals <= cutoff).sum()/len(minqvals) for cutoff in cutoffs])

            if len(minqvals) > 0:
                aucqval = np.trapz([(minqvals <= cutoff).sum()/len(minqvals) for cutoff in cutoffs])/(len(cutoffs)-1)
            else:
                aucqval = 0

            ## auc odds
            newodds = odds.copy()
            newodds.values[(qvals > 0.1).as_matrix().astype(np.bool)] = 0

            aucodds, aucodds1, aucodds2 = cal_faucodds(newodds)

    scores = {}

    if calculate_aucodds:
        scores["aucqval"] = aucqval

        scores["aucodds"] = aucodds
        scores["aucodds1"] = aucodds1
        scores["aucodds2"] = aucodds2

        scores["percenriched"] = (qvals.min(1) < 0.05).sum()/binding.shape[1]

        scores["percenriched_modules"] = (qvals.min(0) < 0.05).sum()/len(modules) if len(modules) > 0 else 0
        scores["percenriched_gsets"] = (qvals.min(1) < 0.05).sum()/binding.shape[1]

        if aucqval > 0 and scores["aucodds"] > 0:
            scores["Fauc"] = 2/(1/scores["aucodds"] + 1/scores["aucqval"])
        else:
            scores["Fauc"] = 0

    scores["bhi"] = bhi

    return scores

def cal_faucodds(odds):
    ## TODO: cutoffs = all odds ratio's (more exact and faster)
    ## TODO: odds and qvals min calculation for at the same context (eg. use the minimal qval to get the odds ratio for the regulator instead of doing it separatedly)
    maxodds = odds.max(1)

    if len(maxodds) == 0:
        return 0

    cutoffs = np.linspace(0, 3, 100)

    stillenriched = [(np.log10(maxodds) >= cutoff).sum()/len(maxodds) for cutoff in cutoffs]
    aucodds = np.trapz(stillenriched, cutoffs) / (cutoffs[-1] - cutoffs[0])
    
    maxodds = odds.max(0)

    cutoffs = np.linspace(0, 3, 100)

    stillenriched = [(np.log10(maxodds) >= cutoff).sum()/len(maxodds) for cutoff in cutoffs]
    aucodds2 = np.trapz(stillenriched, cutoffs) / (cutoffs[-1] - cutoffs[0])
    return 2/(1/aucodds + 1/aucodds2), aucodds, aucodds2

def fmeasure_wiwie(c, k):
    if (len(c) == 0) or (len(k) == 0):
        return 0

    tp = 0
    tn = 0
    fp = 0
    fn = 0
    
    overlaps = []
    for ki in k:
        overlaps_row = []
        for ci in c:
            overlaps_row.append(len(set(ci) & set(ki)))
        ci = c[np.argmax(overlaps_row)]
        
        tp += len(set(ci) & set(ki))
        fp += len(set(ki).difference(set(ci)))
        fn += len(set(ci).difference(set(ki)))
        
        #overlaps.append(overlaps_row)
    #overlaps = np.array(overlaps)
    
    precision = tp/(tp+fp)
    recall = tp/(tp+fn)

    try:
        return 2/(1/precision+1/recall)
    except:
        return 0

def fmeasure_flowcap(c, k):
    if (len(c) == 0) or (len(k) == 0):
        return 0

    G = {g for ci in c for g in ci} | {g for ki in k for g in ki}
    
    precisions = []
    recalls = []
    for ci in c:
        precisions_row = []
        recalls_row = []
        for ki in k:
            precisions_row.append(len(set(ci) & set(ki))/np.float64(len(ki)))
            recalls_row.append(len(set(ci) & set(ki))/np.float64(len(ci)))
        precisions.append(precisions_row)
        recalls.append(recalls_row)
    precisions = np.array(precisions)
    recalls = np.array(recalls)

    F = 2/(1/precisions+1/recalls)
    
    bestF = F.max(0) * np.array([len(ki) for ki in k])/len(G)

    return bestF.sum()

def get_membership(c, G):
    return pd.DataFrame([[g in ci for g in G] for ci in c])

def vmeasure_wiwie(c, k):
    if (len(c) == 0) or (len(k) == 0):
        return 0

    G = {g for ci in c for g in ci} | {g for ki in k for g in ki}
    a = get_membership(c, G).astype(np.int).dot(get_membership(k, G).astype(np.int).T)

    N = a.sum().sum()
    #N = len(G)

    HC = -(a.sum(0)/N * np.log(a.sum(0)/N)).fillna(0).sum()
    HC_K = - (a/N * np.log(a.T/a.sum(1)).T).fillna(0).sum().sum()

    if HC == 0 or HC_K == 0:
        return 0
    h = 1-HC_K/HC

    HK = -(a.sum(1)/N * np.log(a.sum(1)/N)).fillna(0).sum()
    HK_C = - (a/N * np.log(a/a.sum(0))).fillna(0).sum().sum()


    if HK == 0 or HK_C == 0:
        return 0
    o = 1-HK_C/HK

    if h == 0 or o == 0 or (h + o == 0):
        return 0

    V = 2/(1/h+1/o)
    
    if pd.isnull(V):
        V = 0
    
    return V



class ModCviEval:
    def __init__(self, settings_name=None):
        self.settings_name = settings_name
        self.scoring_folder = "../results/modcvieval/"

    def run(self, settings, pool):
        jobs = []
        manager = mp.Manager()
        scores = manager.dict()

        params_pool = []

        i = 0

        self.modulesets = [Modules(json.load(open("../" + setting["output_folder"] + "modules.json"))) for setting in settings]

        print("Evaluating a total of " + str(len(settings)) + " settings.")

        for setting, modules in zip(settings, self.modulesets):

            runinfo = json.load(open("../" + setting["output_folder"] + "runinfo.json"))

            method = json.load(open("../" + setting["method_location"]))
            dataset = json.load(open("../" + setting["dataset_location"]))

            params_pool.append((i, modules, method, dataset, runinfo, None, scores))

            i+=1

        pool.starmap(modcvievalworker, params_pool)

        self.scores = pd.DataFrame(list(scores.values()))
        self.scores_full = list(scores.values())

        manager.shutdown()

    def save(self):
        self.scores.to_csv(self.scoring_folder + self.settings_name + ".tsv", sep="\t")
        if hasattr(self, 'scores_full'):
            json.dump(self.scores_full, open(self.scoring_folder + self.settings_name + ".json", "w"), cls=JSONExtendedEncoder)

    def load(self):
        self.scores = pd.read_table(self.scoring_folder + self.settings_name + ".tsv", index_col=0)

def modcvievalworker(settingid, modules, method, dataset, runinfo, baseline, scores):
    print("▶ " + str(settingid))
    E = pd.read_pickle("../data/expression/" + dataset["datasetname"] + "/E.pkl")
    print(dataset)
    print(E.columns)
    print("--------------------")
    distances = pd.read_pickle("../data/expression/" + dataset["datasetname"] + "/simdist/pearson_correlation_distance.pkl")

    modules = modules.filter_size(5)
    
    Ginmodules = {g for module in modules for g in module}
    Gnotinmodules = [g for g in E.columns if g not in Ginmodules]

    # make one big garbage module?
    #if len(Gnotinmodules) > 0:
        #modules.append(Gnotinmodules)
    # put every unassigned gene in its own module?
    #for g in Gnotinmodules:
        #modules.append([g]) 
    #assign every gene to the module with which it correlates the most
    print("<-- " + str(len(Gnotinmodules)))
    if len(Gnotinmodules) > 0:
        Emod = cal_module_average(modules, E)
        cors = np.corrcoef(E[Gnotinmodules].T, Emod.T)[len(Gnotinmodules):, :-len(modules)].T
        for g, modulecors in zip(Gnotinmodules, cors):
            modulesid = np.argmax(modulecors)
            modules[modulesid].add(g)

    Ginmodules = {g for module in modules for g in module}
    Gnotinmodules = [g for g in E.columns if g not in Ginmodules]
    print("--> " + str(len(Gnotinmodules)))

    clustercenters = cal_clustercenters(modules, E)
    dispersions = cal_dispersions(modules, E, clustercenters)

    settingscores = {}

    for cvi_name in ["asw", "dbindex", "dbstarindex", "ch"]:
        settingscores[cvi_name] = cal_cvi(modules, E, distances, cvi_name, clustercenters, dispersions)

    settingscores.update(method["params"])
    settingscores.update(dataset["params"])

    scores[settingid] = settingscores

    print("◼ " + str(settingid))



class ModNetEval:
    def __init__(self, settings_name=None):
        self.settings_name = settings_name
        self.scoring_folder = "../results/modneteval/"

    def run(self, settings):
        self.modulesets = [Modules(json.load(open("../" + setting["output_folder"] + "modules.json"))) for setting in settings]

        print("Evaluating a total of " + str(len(settings)) + " settings.")

        netcombinations = {}

        scores = []

        for i, (setting, modules) in enumerate(zip(settings, self.modulesets)):
            print('{0:>4}/{1:<8} {2:>8}'.format(str(i), str(len(settings)), str(len(modules))))
            runinfo = json.load(open("../" + setting["output_folder"] + "runinfo.json"))
            method = json.load(open("../" + setting["method_location"]))
            dataset = json.load(open("../" + setting["dataset_location"]))

            ##
            # load known and observed networks if not yet loaded
            if setting["dataset_name"] not in netcombinations.keys():
                netcombinations[setting["dataset_name"]] = load_network_combinations(dataset)
            ##

            for (directni_method, regnet), (wnet, knet, sharedgenes) in netcombinations[setting["dataset_name"]].items():
                wnetobs = infer_directni_modules(wnet, modules, sharedgenes)
                settingscores = score_network(wnetobs, knet)

                settingscores.update(method["params"])
                settingscores.update(dataset["params"])

                settingscores["directni_method"] = directni_method
                settingscores["regnet"] = regnet
                settingscores["runningtime"] = runinfo["runningtime"]

                scores.append(settingscores)

        self.scores = pd.DataFrame(scores)
        self.scores_full = scores

    def save(self):
        self.scores.to_csv(self.scoring_folder + self.settings_name + ".tsv", sep="\t")
        if hasattr(self, 'scores_full'):
            json.dump(self.scores_full, open(self.scoring_folder + self.settings_name + ".json", "w"), cls=JSONExtendedEncoder)

    def load(self):
        self.scores = pd.read_table(self.scoring_folder + self.settings_name + ".tsv", index_col=0)

def load_network_combinations(dataset):
    combinations = {}
    Gmap = pd.read_table("../data/expression/" + dataset["datasetname"] + "/Gmap.tsv", sep="\t", squeeze=True, index_col=0, header=None)
    Gmap.index = Gmap.index.astype(str)

    for directni_method in ["genie3", "clr", "tigress", "correlation"]:
        if dataset["datasetname"] in ["ecoli_colombos", "ecoli_dream5", "yeast_gpl2529", "yeast_dream5"]:
            sep = ","
        else:
            sep = "\t"

        wnet_original = pd.read_table("../results/directni/" + dataset["datasetname"] + "/" + directni_method + ".csv", sep=sep, index_col=0)
        for regnet in dataset["regnets"]:
            try:
                knet = pd.read_pickle("../data/regnets/" + regnet + "/adjacency.pkl")
            except:
                knet = pd.read_table("../data/regnets/" + regnet + "/adjacency.csv", sep="\t", index_col=0)

            print(1)
            knet.index = Gmap[knet.index]
            knet.columns = Gmap[knet.columns]

            print(2)
            sharedregulators = sorted(list(set(knet.columns) & set(wnet_original.columns)))
            sharedgenes = sorted(list(set(knet.index) & set(wnet_original.index)))

            print(3)
            wnet = wnet_original.copy().ix[sharedgenes, sharedregulators]
            knet = knet.ix[sharedgenes, sharedregulators]

            print(4)
            combinations[(directni_method, regnet)] = [wnet, knet, sharedgenes]

            print(len(sharedgenes), len(sharedregulators))
    print("loaded")
    return combinations

def infer_directni_modules(wnet, modules, sharedgenes, bymodules=False):
    """
        wnet: weighted network with genes or modules (see bymodules) in rows and regulators in columns
        modules: list of modules
        sharedgenes: all genes
        bymodules: whether the rows in wnet denote modules (in the same order as modules) or genes
    """
    if bymodules:
        wnetobs = pd.DataFrame(np.random.random([len(sharedgenes), len(wnet.columns)])*0.001, index=sharedgenes, columns=wnet.columns)
    else:
        wnetobs = pd.DataFrame(np.random.random(wnet.shape)*0.001, index=wnet.index, columns=wnet.columns)
    wmnet = []
    
    sharedgenes = set(sharedgenes)

    for moduleid, module in enumerate(modules):
        if len(module) >= 5:
            module = set(module) & sharedgenes
            if bymodules:
                wnet_module = wnet.ix[moduleid]
            else:
                wnet_module = wnet.ix[module].mean()
            wmnet.append(wnet_module)

            # directly
            newweights = np.repeat(wnet_module.as_matrix().reshape(1, wnet_module.shape[0]), len(module), 0)
            wnetobs.ix[module] = np.maximum(wnetobs.ix[module], newweights)

            # gene by gene: slow! (duh)
            #for g in set(module) & sharedgenes:
            #    wnetobs_g = wnetobs.ix[g]
            #    wnetobs.ix[g] = np.maximum(wnetobs_g, wnet_module + np.random.random(wnet_module.shape)*0.001)
    return wnetobs

from sklearn import metrics
def score_network(wnetobs, knet, plotcurve=False):
    known = knet.as_matrix().flatten().astype(np.bool)
    observed = wnetobs.as_matrix().flatten()
    
    prcurve = metrics.precision_recall_curve(known, observed)
    if plotcurve:
        plot(prcurve[0], prcurve[1])
    return {
        "aupr":metrics.average_precision_score(known, observed), 
        "auroc":metrics.roc_auc_score(known, observed),
        "precision_500":prcurve[0][-min(500, len(prcurve[0])-1)],
        "precision_1000":prcurve[0][-min(1000, len(prcurve[0])-1)],
        "precision_2000":prcurve[0][-min(2000, len(prcurve[0])-1)],
    }