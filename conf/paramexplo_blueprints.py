import numpy as np

blueprints = {
"flame": {
    "staticparams":{
        "method":"flame",
        "threshold2": -3
    },
    "dynparams": {
        "knn":[1, 2, 3, 4, 5, 7, 9, 11, 13],
        "threshold":[-1, 0.01, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 0.95, 0.99]
    },
    "type":"moduledetection"
},
"kmedoids": {
    "staticparams":{
        "method":"kmedoids",
        "simdist_function":"pearson_correlation"
    },
    "dynparams": {
        "k":np.arange(25, 300.01, 25)
    },
    "type":"moduledetection"
},
"som": {
    "staticparams":{
        "method":"som"
    },
    "dynparams": {
        "dim":np.arange(6, 27.01, 3),
        "radius":[0.5, 1, 1.5, 2],
        "topo":["rectangular", "hexagonal"]
    },
    "type":"moduledetection"
},
"kmeans": {
    "staticparams":{
        "method":"kmeans"
    },
    "dynparams": {
        "k":np.arange(25, 300.01, 25)
    },
    "type":"moduledetection"
},
"cmeans": {
    "staticparams":{
        "method":"cmeans"
    },
    "dynparams": {
        "k":np.arange(25, 325.01, 50),
        "m":[1.01, 1.02, 1.05, 1.1],
        "cutoff":[0.001, 0.01, 0.05, 0.1, 0.2, 0.5]
    },
    "type":"moduledetection"
},
"spectral_similarity": {
    "staticparams":{
        "method":"spectral_similarity",
        "simdist_function":"pearson_correlation"
    },
    "dynparams": {
        "k":np.arange(25, 300.01, 25)
    },
    "type":"moduledetection"
},
"affinity": {
    "staticparams":{
        "method":"affinity",
        "simdist_function": "pearson_correlation"
    },
    "dynparams": {
        "preference_fraction":np.arange(-3, 1.01, 0.25)
    },
    "type":"moduledetection"
},
"spectral_knn": {
    "staticparams":{
        "method":"spectral_knn"
    },
    "dynparams": {
        "k":np.arange(25, 300.01, 25),
        "knn":[10, 20, 30, 50, 70, 100]
    },
    "type":"moduledetection"
},
"wgcna": {
    "staticparams":{
        "method":"wgcna",
        "power": 6,
        "mergeCutHeight": 0.15,
        "minModuleSize": 5,
        "deepSplit": 2,
        "detectCutHeight": 0.995,
        "TOMDenom": "min"
    },
    "dynparams": {
        "power":np.arange(1, 10.01, 1),
        "mergeCutHeight": np.arange(0.05, 0.501, 0.05)
    },
    "type":"moduledetection"
},
"agglom": {
    "staticparams":{
        "method":"agglom",
        "simdist_function":"pearson_correlation"
    },
    "dynparams": {
        "linkage":["complete", "average"],
        "k":np.arange(25, 300.01, 25)
    },
    "type":"moduledetection"
},
"hybrid": {
    "staticparams":{
        "method":"hybrid"
    },
    "dynparams": {
        "k":np.arange(25, 300.01, 25)
    },
    "type":"moduledetection"
},
"divisive": {
    "staticparams":{
        "method":"divisive"
    },
    "dynparams": {
        "k":np.arange(25, 300.01, 25)
    },
    "type":"moduledetection"
},
"sota": {
    "staticparams":{
        "method":"sota"
    },
    "dynparams": {
        "alpha":[0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    },
    "type":"moduledetection"
},
"dclust": {
    "staticparams":{
        "method":"dclust",
        "dismethod": "1-cor"
    },
    "dynparams": {
        "delta":np.arange(0.1, 0.81, 0.1),
        "rho":[0.5] + list(np.arange(1, 15.01, 1))
    },
    "type":"moduledetection"
},
"click": {
    "staticparams":{
        "method":"click"
    },
    "dynparams": {
        "homogeneity":np.arange(0, 1.01, 0.05)
    },
    "type":"moduledetection"
},
"dbscan": {
    "staticparams":{
        "method":"dbscan"
    },
    "dynparams": {
        "eps":np.arange(0.05, 0.61, 0.05),
        "MinPts":np.arange(1, 10.01, 1)
    },
    "type":"moduledetection"
},
"meanshift": {
    "staticparams":{
        "method":"meanshift",
        "cluster_all": 1
    },
    "dynparams": {
        "bandwidth":["auto"] + list(np.arange(2.5, 70.01, 2.5))
    },
    "type":"moduledetection"
},
"clues": {
    "staticparams":{
        "method":"clues"
    },
    "dynparams": {
    },
    "type":"moduledetection"
},
"transitivity": {
    "staticparams":{
        "method":"transitivity"
    },
    "dynparams": {
        "threshold":np.arange(-0.5, 1, 0.1),
        "cutoff":[-1, 0.001, 0.01, 0.05, 0.1, 0.2, 0.5]
    },
    "type":"moduledetection"
},
"mcl": {
    "staticparams":{
        "method":"mcl"
    },
    "dynparams": {
        "inflation":[1.4, 2, 3, 4, 6, 8, 10, 15, 20],
        "threshold":np.arange(0, 1, 0.1),
    },
    "type":"moduledetection"
},

## decomposition
"ica_fdr": {
    "staticparams":{
        "method":"ica_fdr"
    },
    "dynparams": {
        "k":np.arange(50, 600.01, 50),
        "qvalcutoff":10**(-np.arange(1, 15, dtype=np.float))
    },
    "type":"moduledetection"
},
"ica_fdr_signed": {
    "staticparams":{
        "method":"ica_fdr_signed"
    },
    "dynparams": {
        "k":np.arange(50, 600.01, 50),
        "qvalcutoff":10**(-np.arange(1, 15, dtype=np.float))
    },
    "type":"moduledetection"
},
"ica_zscore": {
    "staticparams":{
        "method":"ica_zscore"
    },
    "dynparams": {
        "k":np.arange(50, 600.01, 50),
        "stdcutoff":np.arange(0.5, 7.01, 0.5)
    },
    "type":"moduledetection"
},
"ica_max": {
    "staticparams":{
        "method":"ica_max"
    },
    "dynparams": {
        "k":np.arange(50, 600.01, 50),
    },
    "type":"moduledetection"
},
"ipca": {
    "staticparams":{
        "method":"ipca"
    },
    "dynparams": {
        "k":np.arange(50, 600.01, 50),
        "qvalcutoff":10**(-np.arange(1, 15, dtype=np.float))
    },
    "type":"moduledetection"
},
"pca": {
    "staticparams":{
        "method":"pca"
    },
    "dynparams": {
        "k":np.arange(25, 300.01, 25),
        "qvalcutoff":10**(-np.arange(1, 15, dtype=np.float))
    },
    "type":"moduledetection"
},
"nmf_max": {
    "staticparams":{
        "method":"nmf_max"
    },
    "dynparams": {
        "k":np.arange(25, 300.01, 25)
    },
    "type":"moduledetection"
},
"nmf_tail": {
    "staticparams":{
        "method":"nmf_tail"
    },
    "dynparams": {
        "k":np.arange(25, 300.01, 25),
        "tailcutoff":[0.1,0.05,0.02,0.01,0.005,0.002,0.001,0.0005]
    },
    "type":"moduledetection"
},


## biclustering

"spectral_biclust": {
    "staticparams":{
        "method":"spectral_biclust"
    },
    "dynparams": {
        "n":[10, 20, 50, 100, 200, 300, 400, 500],
        "ngenes":[10, 20, 50, 100, 200, 300, 400, 500]
    },
    "type":"moduledetection"
},
"isa": {
    "staticparams":{
        "method":"isa"
    },
    "dynparams": {
        "thr_col":np.arange(0.5, 5.01, 0.5),
        "thr_row":np.arange(0.5, 3.01, 0.5)
    },
    "type":"moduledetection"
},
"biforce": {
    "staticparams":{
        "method":"biforce"
    },
    "dynparams": {
        "t":[0, 0.1, 0.2, 0.5, 0.75, 1, 1.5, 2, 5, 10],
        "m":["o", "u", "l", "h"]
    },
    "type":"moduledetection"
},
"qubic": {
    "staticparams":{
        "method":"qubic"
    },
    "dynparams": {
        "q":np.arange(0.01, 0.511, 0.05),
        "tolerance":np.arange(0.3, 1.01, 0.1)
    },
    "type":"moduledetection"
},
"fabia": {
    "staticparams":{
        "method":"fabia"
    },
    "dynparams": {
        "n":np.arange(25, 300.01, 25),
        "thresZ":[0.05, 0.2, 0.35, 0.5, 0.65],
        "thresL":["None", 0.05, 0.2, 0.35, 0.5, 0.65]
    },
    "type":"moduledetection"
},
"msbe": {
    "staticparams":{
        "method":"msbe"
    },
    "dynparams": {
        "alpha":[0.1, 0.2, 0.3, 0.4],
        "beta":[0, 2, 0.4, 0.6],
        "gamma":[0.5, 0.8, 1.1, 1.4]
    },
    "type":"moduledetection"
},
"plaid": {
    "staticparams":{
        "method":"plaid"
    },
    "dynparams": {
        "col_release":np.arange(0, 1.01, 0.25),
        "row_release":np.arange(0, 1.01, 0.25),
        "max_layers":np.arange(50, 500.01, 50)
    },
    "type":"moduledetection"
},
"opsm": {
    "staticparams":{
        "method":"opsm"
    },
    "dynparams": {
        "l":[1, 5, 10, 15, 20]
    },
    "type":"moduledetection"
},
"chengchurch": {
    "staticparams":{
        "method":"chengchurch"
    },
    "dynparams": {
        "delta":[0.0001, 0.0002, 0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1],
        "alpha":[1.01, 1.1, 1.5]
    },
    "type":"moduledetection"
},
"blockcluster": {
    "staticparams":{
        "method":"blockcluster"
    },
    "dynparams": {
        "ngenes":np.arange(25, 325.01, 50).astype(int),
        "nconditions":np.arange(5, 50, 5),
    },
    "type":"moduledetection"
},
"graphclust": {
    "staticparams":{
        "mingenes":5,
        "merge_overlapping": 1,
        "maxoverlap": 0.8
    },
    "dynparams": {
        "method":["strict", "minimal", "mcl1","mcl2","mcl3","ap1","ap2","ap3","tc1","tc2","tc3"],
        "cutoff":[0.0001, 0.0002, 0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2]
    },
    "type":"moduledetection"
},
"genomica": {
    "staticparams":{
        "method":"genomica"
    },
    "dynparams": {
        "n":np.arange(25, 300.01, 25)
    },
    "type":"moduleni"
},
"merlin": {
    "staticparams":{
        "method":"merlin"
    },
    "dynparams":{
        "h": [0.5,0.6,0.7,0.8],
        "p": [-10, -8, -5, -3],
        "r": [2,4,6,8]
    },
    "type":"moduleni"
},
"baseline_permuted": {
    "staticparams":{
        "method":"baseline_permuted"
    },
    "type":"moduledetection"
},
"baseline_sticky": {
    "staticparams":{
        "method":"baseline_sticky"
    },
    "type":"moduledetection"
},
"baseline_scalefree": {
    "staticparams":{
        "method":"baseline_scalefree"
    },
    "type":"moduledetection"
},

## direct ni
"genie3":{
    "staticparams":{
        "method":"genie3",
        "numcores":24
    },
    "type":"directni"
},
"tigress":{
    "staticparams":{
        "method":"tigress",
        "numcores":24
    },
    "type":"directni"
},
"clr":{
    "staticparams":{
        "method":"clr"
    },
    "type":"directni"
},
"correlation":{
    "staticparams":{
        "method":"correlation"
    },
    "type":"directni"
},
"dummy":{
    "staticparams":{
        "method":"dummy"
    },
    "dynparams":{
        "n":[50, 100, 150, 200]
    },
    "type":"moduledetection"
}

}

for blueprint in blueprints.values():
    if "dynparams" not in blueprint.keys():
        blueprint["dynparams"] = {}

# what parameters does every method have?
methodparamsoi = {
    "flame":["knn", "threshold"],
    "kmedoids":["k", "simdist_function"],
    "som":["dim", "radius", "topo"],
    "kmeans":["k"],
    "cmeans":["k", "m", "cutoff"],
    "spectral_similarity":["k", "simdist_function"],
    "affinity":["preference_fraction", "simdist_function"],
    "spectral_knn":["k", "knn"],
    "wgcna":["power", "mergeCutHeight", "minModuleSize", "deepSplit", "detectCutHeight", "TOMDenom"],
    "agglom":["k", "linkage", "simdist_function"],
    "hybrid":["k"],
    "divisive":["k"],
    "sota":["alpha"],
    "dclust":["delta", "rho"],
    "click":["homogeneity"],
    "dbscan":["eps", "MinPts"],
    "meanshift":["bandwidth"],
    "clues":[],
    "transitivity":["threshold", "cutoff"],
    "mcl":["threshold", "inflation"],

    "ica_fdr":["k", "qvalcutoff"],
    "ica_fdr_signed":["k", "qvalcutoff"],
    "ica_zscore":["k", "stdcutoff"],
    "ipca":["k", "qvalcutoff"],
    "ica_max":["k"],
    "pca":["k", "qvalcutoff"],
    "nmf_max":["k"],
    "nmf_tail":["k", "tailcutoff"],

    "spectral_biclust":["n", "ngenes"],
    "isa":["thr_col", "thr_row"],
    "biforce":["t", "m"],
    "qubic":["q", "tolerance"],
    "fabia":["n", "thresZ", "thresL"],
    "msbe":["alpha", "beta", "gamma"],
    "plaid":["col_release", "row_release", "max_layers"],
    "opsm":["l"],
    "chengchurch":["alpha", "delta"],
    "blockcluster":["ngenes", "nconditions"],

    "genomica":["n"],
    "merlin":["h", "p", "r"],

    "graphclust_genie3":["method", "cutoff", "maxoverlap"],
    "graphclust_clr":["method", "cutoff", "maxoverlap"],
    "graphclust_tigress":["method", "cutoff", "maxoverlap"],
    "graphclust_correlation":["method", "cutoff", "maxoverlap"],

    "baseline_permuted":[],
    "baseline_sticky":[],
    "baseline_scalefree":[],

    "dummy":["n"]
}

methodparams_modulenumber = {
    "flame":["knn"],
    "kmedoids":["k"],
    "som":["dim"],
    "kmeans":["k"],
    "cmeans":["k"],
    "spectral_similarity":["k"],
    "affinity":["preference_fraction"],
    "spectral_knn":["k"],
    "wgcna":["mergeCutHeight"],
    "agglom":["k"],
    "hybrid":["k"],
    "divisive":["k"],
    "sota":["alpha"],
    "dclust":["delta", "rho"],
    "click":["homogeneity"],
    "dbscan":["eps", "MinPts"],
    "meanshift":["bandwidth"],
    "clues":[],
    "transitivity":["threshold"],
    "mcl":["threshold", "inflation"],

    "ica_fdr":["k"],
    "ica_fdr_signed":["k"],
    "ica_zscore":["k"],
    "ipca":["k"],
    "ica_max":["k"],
    "pca":["k"],
    "nmf_max":["k"],
    "nmf_tail":["k"],

    "spectral_biclust":["n", "ngenes"],
    "isa":["thr_col", "thr_row"],
    "biforce":["t"],
    "qubic":["tolerance"],
    "fabia":["n"],
    "msbe":[],
    "plaid":["max_layers"],
    "opsm":["l"],
    "chengchurch":["delta"],

    "genomica":["n"],
    "merlin":["h"],

    "graphclust_genie3":["cutoff"],
    "graphclust_clr":["cutoff"],
    "graphclust_tigress":["cutoff"],
    "graphclust_correlation":["cutoff"],

    "baseline_permuted":[],
    "baseline_sticky":[],
    "baseline_scalefree":[]
}

for method in methodparams_modulenumber:
    methodparamsoi[method + "_auto"] = [param for param in methodparamsoi[method] if param not in methodparams_modulenumber[method]]
    methodparamsoi[method + "_auto"].append("cvi")
