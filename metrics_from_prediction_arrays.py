import csv, sys
try:
    import ujson as json
except ImportError:
    import json
csv.field_size_limit(sys.maxsize)
import numpy as np
from scipy.sparse import csr_matrix, lil_matrix
from sklearn.metrics import precision_recall_fscore_support

from tqdm import tqdm

from xopen import xopen

# Arguments: labels.json predictions.txt.gz test-processed.gz ontology_files*

with xopen(sys.argv[1]) as f:
    labels = json.load(f)

with xopen(sys.argv[3]) as f:
    cr = csv.reader(f, delimiter="\t")
    example_num = int(next(cr)[0]) # Read example number row.
    labels_true = lil_matrix((example_num, len(labels)), dtype='b')
    for i, line in tqdm(enumerate(cr), desc="Loading test data"):
        labels_true[i] = json.loads(line[1])
    labels_true = csr_matrix(labels_true)
    #print(labels_true[0])

with xopen(sys.argv[2]) as f:
    cr = csv.reader(f)
    scores_pred = np.ndarray(labels_true.shape)
    for i, line in tqdm(enumerate(cr), desc="Loading predictions"):
        scores_pred[i] = json.loads(line[0])
    #scores_pred = csr_matrix(scores_pred)
    #print(labels_true[0])

ontologies = {}
for f_name in sys.argv[4:]:
    ontology = set()
    with xopen(f_name) as f:
        for line in f:
            ontology.add(line.strip())
    ontologies[f_name.split('.')[0].split('/')[-1]] = ontology

# ontology_indices = {}
output_scores = {}
output_labels_true = {}
print("num_ontologies:", len(ontologies))
for o in ontologies:
    ontology_index = []
    for i, label in enumerate(labels):
        if label in ontologies[o]:
            ontology_index.append(i)
    with xopen(o + "_index.json", "wt") as f:
        json.dump(ontology_index, f)
    output_scores[o] = np.ndarray((example_num, len(ontologies[o])))
    output_scores[o] = scores_pred[:,ontology_index]
    output_labels_true[o] = lil_matrix((example_num, len(ontologies[o])), dtype='b')
    output_labels_true[o] = labels_true[:,ontology_index]

# TODO: do things with numpy/scipy arrays instead in this code.

# for i in tqdm(range(example_num), desc="Splitting ontologies"):
#     for j, label in enumerate(labels):
#         for o in ontologies:
#             if label in ontologies[o]:
#                 output_scores[o][i,j] = scores_pred[i,j]
#                 output_labels_true[o][i,j] = labels_true[i,j]

for o in output_scores:
    print("Ontology", o, "num predictions:", output_scores[o].shape[1])
    print("Ontology", o, "num true:", output_labels_true[o].shape[1])

for o in output_scores:
    print(o)

    labels_prob = output_scores[o]
    labels_true = output_labels_true[o]

    for threshold in np.arange(0, 1.05, 0.1):
        print("Threshold:", threshold)
        labels_pred = lil_matrix(output_scores[o].shape, dtype='b')
        labels_pred[labels_prob>=threshold] = 1
        precision, recall, f1, _ = precision_recall_fscore_support(labels_true, labels_pred, average="micro")
        print("Precision:", precision)
        print("Recall:", recall)
        print("F1-score:", f1, "\n")