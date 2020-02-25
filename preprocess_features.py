import csv, sys
csv.field_size_limit(sys.maxsize)

import numpy as np

from math import log
from tqdm import tqdm
from xopen import xopen

try:
    import ujson as json
except ImportError:
    import json

with xopen(sys.argv[2]) as label_file:
    labels = json.load(label_file)

label_to_index = {label:i for i, label in enumerate(labels)}

GO_errors = set()

with xopen(sys.argv[1]) as features_file, xopen(sys.argv[3], "wt") as out_file:
    cr = csv.reader(features_file, delimiter='\t')
    for line in tqdm(cr, desc="Processing features"):
        out_line = np.zeros(len(labels), dtype='float32')
        # Get rid of lines that are just "\n".
        if not line:
            out_file.write(json.dumps(out_line.tolist()) + '\n')
            continue
        # # Get rid of lines that are "\t\t\n", because that also happens.
        if not line[0].strip():
            out_file.write(json.dumps(out_line.tolist()) + '\n')
            continue
        terms = line[0].split(',')
        # IDENTITY % IF LINE[1]!!!
        bit_scores = line[2].split(',')
        for term, bit_score in zip(terms, bit_scores):
            bit_score = log(float(bit_score), 2)
            if bit_score < 7.6:
                bit_score = 0.0
            try:
                out_line[label_to_index[term]] = bit_score
            # Labels not in the training data.
            except KeyError:
                GO_errors.add(term)
        
        out_file.write(json.dumps(out_line.tolist()) + '\n')

print("Terms that KeyErrored:", GO_errors)