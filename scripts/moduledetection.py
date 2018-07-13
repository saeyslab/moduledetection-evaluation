import sys
import os
sys.path.insert(0,os.path.abspath("lib/"))

import json
import shutil

from util import JSONExtendedEncoder

from clustering import *

import pandas as pd
import numpy as np

import time

import traceback

method = json.load(open(sys.argv[1]))
dataset = json.load(open(sys.argv[2]))
output_folder = sys.argv[3]
if len(sys.argv) > 4:
	originaltime = int(sys.argv[4])
else:
	originaltime = 0

if dataset["expression"].endswith(".pkl"):
	E = pd.read_pickle(dataset["expression"])
elif dataset["expression"].endswith(".hdf"):
	E = pd.read_hdf(dataset["expression"], "E")
else:
	E = pd.read_csv(dataset["expression"], sep="\t", index_col=0)
if len(sys.argv) > 5:
	if sys.argv[5] == "test":
		E = E.ix[E.index[1:100], E.columns[1:100]]

starttime = time.time() - originaltime

try:
	modules = locals()[method["params"]["method"]](E, simdistfolder=dataset["simdistfolder"],  **method["params"])
except BaseException as e:
	print("Error during clustering: ")
	print(e)
	traceback.print_exc()
	modules = []

endtime = time.time()

if os.path.exists(output_folder):
	shutil.rmtree(output_folder)
os.makedirs(output_folder)
json.dump(modules, open(output_folder + "/modules.json", "w"), cls=JSONExtendedEncoder)
json.dump({"runningtime":endtime-starttime}, open(output_folder + "/runinfo.json", "w"), cls=JSONExtendedEncoder)
json.dump(method["params"], open(output_folder + "method.json", "w"))
