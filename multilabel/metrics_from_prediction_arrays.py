import csv, sys
import ujson as json
csv.field_size_limit(sys.maxsize)
import numpy as np
from scipy.sparse import lil_matrix
from sklearn.metrics import precision_recall_fscore_support

from tqdm import tqdm

from xopen import xopen

with xopen(sys.argv[1]) as f:
    labels = json.load(f)

with xopen(sys.argv[3]) as f:
    labels_true = []
    cr = csv.reader(f, delimiter="\t")
    next(cr) # Skip example number row.
    for line in tqdm(cr, desc="Loading test data"):
        labels_true.append(json.loads(line[1]))
    labels_true = np.array(labels_true, dtype='b')
    print(labels_true[0])

ontologies = {}
for f_name in sys.argv[4:]:
    ontologies[f_name.split('.')[0]] = set()
    with xopen(f_name) as f:
        for line in f:
            ontologies[f_name.split('.')[0]].add(line.strip())

output_scores = {}
output_scores_true = {}
print("num_ontologies:", len(ontologies))
for k in ontologies:
    output_scores[k] = []
    output_scores_true[k] = []

# TODO: do things with numpy/scipy arrays instead in this code.

with xopen(sys.argv[2]) as predictions:
    cr = csv.reader(predictions)
    for line, line_true in tqdm(zip(cr, labels_true), desc="Parsing predictions"):
        #print(line_true)
        output_lines = {}
        output_lines_true = {}
        for k in ontologies:
            output_lines[k] = []
            output_lines_true[k] = []
        for score, label, label_true in zip(json.loads(line[0]), labels, line_true):
            # print(label_true); input()
            for k in ontologies:
                if label in ontologies[k]:
                    output_lines[k].append(score)
                    output_lines_true[k].append(label_true)
        for k in ontologies:
            output_scores[k].append(output_lines[k][:])
            output_scores_true[k].append(output_lines_true[k][:])

for k in output_scores:
    print("Ontology", k, "num predictions:", len(output_scores[k][0]))
    print("Ontology", k, "num true:", len(output_scores_true[k][0]))

for o in output_scores:
    print(o)
    labels_prob = lil_matrix(output_scores[o])
    print(labels_prob.shape)
    print(labels_prob[0])
    labels_true = lil_matrix(output_scores_true[o])

    for threshold in np.arange(0.1, 1.0, 0.1):
        print("Threshold:", threshold)
        labels_pred = lil_matrix(labels_prob.shape, dtype='b')
        labels_pred[labels_prob>threshold] = 1
        precision, recall, f1, _ = precision_recall_fscore_support(labels_true, labels_pred, average="micro")
        print("Precision:", precision)
        print("Recall:", recall)
        print("F1-score:", f1, "\n")