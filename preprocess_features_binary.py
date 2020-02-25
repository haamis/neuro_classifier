import csv, sys
csv.field_size_limit(sys.maxsize)

import numpy as np

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

with xopen(sys.argv[1]) as features_file, xopen("features_out.gz", "wt") as out_file:
    cr = csv.reader(features_file, delimiter='\t')
    for line in tqdm(cr, desc="Processing features"):
        out_line = np.zeros(len(labels), dtype='b')
        # Get rid of lines that are just "\n".
        if not line:
            out_file.write(json.dumps(out_line.tolist()) + '\n')
            continue
        terms = line[0].split(',')
        for term in terms:
            try:
                out_line[label_to_index[term]] = 1
            # Labels not in the training data.
            except KeyError:
                GO_errors.add(term)
        
        out_file.write(json.dumps(out_line.tolist()) + '\n')

print("Terms that KeyErrored:", GO_errors)