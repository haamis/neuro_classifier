import csv, sys

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
#print("l2i", label_to_index)

out_lines = []

with xopen(sys.argv[1]) as features_file:
    for line in tqdm(features_file, desc="Processing features"):
        line = line.strip()
        if not line:
            out_lines.append([])
            continue
        out_line = np.zeros(len(labels), dtype='b')
        terms = line.split(',')
        for term in terms:
            try:
                out_line[label_to_index[term]] = 1
            # Ignore labels not in the training data for now.
            except KeyError:
                pass
        out_lines.append(out_line.tolist())

print(len(out_lines))

with xopen("features_out.gz", "wt") as out_file:
    for line in tqdm(out_lines, desc="Writing output"):
        out_file.write(json.dumps(line) + '\n')