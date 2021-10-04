import numpy as np
import pandas as pd

from sklearn.preprocessing import scale

from modulecontainers import Module, Modules

from util import TemporaryDirectory

import os
import subprocess as sp

from collections import defaultdict

from clustering import agglom, convert_modules2labels, convert_labels2modules

def genomica(E, R, n=100, **kwargs):
    E_genomica = pd.DataFrame(np.vstack([np.array([["desc" for i in range(len(E.columns))], ["desc" for i in range(len(E.columns))]]), scale(E)]), index=["desc", "desc"]  + E.index.tolist(), columns=E.columns).T
    E_genomica.index.name = "genes"

    with TemporaryDirectory() as tmpdir:
        E_genomica.to_csv(tmpdir + "/E.csv", sep="\t")
        
        R_genomica = {gid:g for gid, g in enumerate(E_genomica.index) if g in R and g in E.columns}
        with open(tmpdir + "/regulators.csv", "w") as outfile:
            outfile.write("\n".join([str(gid) for gid in R_genomica.keys()]))
        
        # PERSOFTWARELOCATION is the location in which the software is installed
        genomica_loc = os.environ["PERSOFTWARELOCATION"] + "/Genomica/new/"
        genomica_command = "cd {genomica_loc};java -XX:ParallelGCThreads=1 -Xmx12G -cp .:../Genomica.jar ExampleProgram ".format(**locals())
        
        n = int(n)

        command = genomica_command + "{tmpdir}/E.csv {tmpdir}/regulators.csv {n} 10 > {tmpdir}/output.txt".format(**locals())
        
        sp.call(command, shell=True)
        
        # postprocess output file
        state = "members"
        modules = []
        modulenet = []
        with open(tmpdir + "/output.txt") as infile:
            for line in infile.readlines():
                line = line.rstrip()

                if line.startswith("Module members: "):
                    moduleregulator_scores = defaultdict(float)
                    
                    state = "members"
                    module = Module([E.columns[int(geneid)] for geneid in line[len("Module members: "):].split(" ")])
                    print("----")
                    print(module)
                    print(len(modules))
                elif line.startswith("<<<<<PROGRAM"):
                    state = "program"
                elif line.find("Regulator: ") > -1:
                    
                    line = line.split(",")

                    regulatorid = int(line[0].split(" ")[-1])
                    #print(line[0].split(" ")[-1])

                    #print(R_genomica[regulatorid], regulatorid, float(line[-1].split(" ")[-1]))

                    moduleregulator_scores[R_genomica[regulatorid]] += float(line[-1].split(" ")[-1])
                elif line.startswith(">>>>>MODULE"):
                    #print(moduleregulator_scores)
                    
                    modules.append(module)
                    modulenet.append(moduleregulator_scores)

    modulenet = pd.DataFrame(modulenet, columns=R).fillna(0)

    return modules, modulenet

def merlin(E,R, h=0.6, p=5, r=4, initial_number=300, **kwargs):
    if E.min().min() <= 0:
        E = E + np.abs(E.min().min()) + 0.1

        print(E.min().min())

    R = sorted(list(set(R) & set(E.columns)))

    with TemporaryDirectory() as tmpdir:
        E.to_csv(tmpdir + "/E.csv", sep="\t", index=False)
        with open(tmpdir + "/R.csv", "w") as outfile:
            outfile.write("\n".join(R))

        modules = agglom(E, number=initial_number)
        labels = convert_modules2labels(modules, E.columns)
        with open(tmpdir + "/clusterassign.csv", "w") as outfile:
            outfile.write("\n".join([g + "\t" + str(label) for g,label in labels.items()]))
        
        # PERSOFTWARELOCATION is the location in which the software is installed
        binary = os.environ["PERSOFTWARELOCATION"] + "/gpdream/modules/Merlin/src/merlin"
        
        command = "{binary} -d {tmpdir}/E.csv -o {tmpdir} -l {tmpdir}/R.csv -c {tmpdir}/clusterassign.csv -v 1 -h {h} -k 300 -p {p} -r {r}".format(**locals())
        
        print(command)
        
        sp.call(command, shell=True)
        
        labels = pd.read_csv(tmpdir + "/fold0/modules.txt", sep="\t", squeeze=True, index_col=0, header=None)
        modules = convert_labels2modules(labels, labels.index)
        print(labels)
        
        netable = pd.read_csv(tmpdir + "/fold0/prediction_k300.txt", sep="\t", names=["regulator", "target", "score"])
        
        wnet = pd.DataFrame(0, columns=R, index=E.columns)
        for i, (regulator, target, score) in netable.iterrows():
            wnet.ix[target, regulator] = score

    return modules, wnet