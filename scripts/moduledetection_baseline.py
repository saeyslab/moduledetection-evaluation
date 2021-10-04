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
output_folder = sys.argv[5]
regnet_name = sys.argv[3]
knownmodules_name = sys.argv[4]

knownmodules = json.load(open(dataset["knownmodules"][regnet_name][knownmodules_name]))

starttime = time.time()

try:
	modules = locals()[method["params"]["method"]](knownmodules, simdistfolder=dataset["simdistfolder"],  **method["params"])
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
