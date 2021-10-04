import pandas as pd
import numpy as np
class Module(set):
    def __init__(self, genes=None):
        if genes is not None:
            set.__init__(self, genes)

    def filter_retaingenes(self, retaingenes):
        return(Module(set(self) & set(retaingenes)))

    def to_json(self):
        return "[" + ", ".join(sorted(list(self)))

class Modules(list):
    def __init__(self, modules=None):
        if modules is not None:
            # coerce arguments to Module
            list.__init__(self, [Module(module) for module in modules])

    def filter_retaingenes(self, retaingenes):
        return Modules([module.filter_retaingenes(retaingenes) for module in self])

    def filter_size(self, minsize):
        return Modules([module for module in self if len(module) >= minsize])

    def cal_membership(self, G=None):
        if G is None:
            G = sorted(list(set([g for module in self for g in module])))

        membership = []
        for module in self:
            membership.append([g in module for g in G])

        if len(self) > 0:
            return pd.DataFrame(np.array(membership, ndmin=2), columns=G, index=["M" + str(i) for i in range(len(self))]).T
        else:
            return pd.DataFrame(np.zeros((0, len(G))), columns=G).T

    def cal_membership(self, G=None):
        if G is None:
            G = sorted(list(set([g for module in self for g in module])))
        g2id = {g:i for i,g in enumerate(G)}

        membership = np.zeros((len(G), len(self)))
        for i, module in enumerate(self):
            membership[[g2id[g] for g in module if g in g2id.keys()], i] = 1

        if len(self) > 0:
            return pd.DataFrame(membership, columns=["M" + str(i) for i in range(len(self))], index=G)
        else:
            return pd.DataFrame(np.zeros((0, len(G))), columns=G).T

    def shuffle(self, G=None, perc=None):
        """
        "Shuffling" of a modules while keeping the internal structure (size, number and overlap) the same.
        Will map the gene list to a permuted gene list.

        If G is not given, will infer G from all genes within the modules
        """

        if G is None:
            G = list({gene for module in self for gene in module})

        membership = self.cal_membership(G)

        if perc is None or perc == 1:
            genemap = np.random.permutation(G)
        else:
            chosenids = np.random.choice(np.arange(len(G)), int(perc * len(G)), False)

            chosen = [G[i] for i in chosenids]
            chosen = np.random.permutation(chosen).tolist()
            genemap = np.array([G[i] if i not in chosenids else chosen.pop(0) for i in range(len(G))])
        
        shuffledmodules = Modules([genemap[np.where(row)] for row in membership.T.values])

        return shuffledmodules

    def cal_connectivity(self, G=None):
        if G is None:
            G = sorted(list(set([g for module in self for g in module])))
        connectivity = pd.DataFrame(0, index=G, columns=G, dtype=np.bool)
        for module in self:
            connectivity.ix[module, module] = True
        return connectivity


class Bicluster(Module):
    def __init__(self, genes, conditions):
        Module.__init__(self, genes)
        self.genes = self
        self.conditions = conditions

    def __repr__(self):
        return "Bicluster([" + ",".join(["'" + g + "'" for g in self]) + "], [" +",".join(["'" + c + "'" for c in self.conditions]) + "])"

    def size(self):
        return len(self.genes) * len(self.conditions)