import sys, csv
try:
    import ujson as json
except ImportError:
    import json
csv.field_size_limit(sys.maxsize)

import networkx

import numpy as np

from functools import lru_cache, partial

from multiprocessing import Pool

from tqdm import tqdm
from xopen import xopen

def propagate(example, graph):
    for label_index, label_score in enumerate(example):
        # print(label_score)
        if label_score > 0.0:
            # try:
            # print(networkx.ancestors(graph, label_index))
            # print(graph.predecessors(label_index))
            for parent in networkx.ancestors(graph, label_index):
                if example[parent] < label_score:
                    # print("set score for", parent)
                    example[parent] = label_score
            # except:
                # pass

    return example

with open(sys.argv[1]) as f:
    label_names = json.load(f)
    label_names = np.asarray(label_names)

label_to_index = {k:v for v,k in enumerate(label_names)}

graph = networkx.DiGraph()

with xopen("/mnt/extra_raid/sukaew/CAFA4/data/CAFA4_GO/data/parent_term2term.txt.gz") as f:
    cr = csv.reader(f, delimiter='\t')
    next(cr) # skip header
    for line in cr:
        try:
        #parent_dict[label_to_index[line[1]]].append(label_to_index[line[0]])
            graph.add_edge(label_to_index[line[0]], label_to_index[line[1]]) # Order: parent, child
        except KeyError: # Label not found in training data.
            pass

with xopen(sys.argv[2]) as f:
    labels_prob = []
    for line in tqdm(f, desc="Loading predictions"):
        labels_prob.append(np.asarray(json.loads(line)))

labels_prob = np.asarray(labels_prob)
print(labels_prob.shape)

print("Propagating score..")
with Pool(16) as p:
    labels_prob = p.map(partial(propagate, graph=graph), labels_prob)

# print(labels_prob)

labels_prob = np.asarray(labels_prob)

with xopen("predict_output.gz", "wt") as f:
    for prob in tqdm(labels_prob, desc="Writing predict_output.gz"):
        f.write(json.dumps(prob.tolist()) + '\n')