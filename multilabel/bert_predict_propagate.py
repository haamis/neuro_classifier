import sys, json, csv

import tensorflow as tf
import keras.backend as K
import numpy as np

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from math import ceil

from xopen import xopen
from tqdm import tqdm

from functools import lru_cache

from scipy.sparse import lil_matrix
from sklearn.metrics import precision_recall_fscore_support

from keras.backend.tensorflow_backend import set_session
from keras.models import load_model

from keras_bert import get_custom_objects

# Read example count from the first row a preprocessed file.
def get_example_count(file_path):
    with xopen(file_path, "rt") as f:
        cr = csv.reader(f, delimiter="\t")
        return int(next(cr)[0])

def data_generator(file_path, batch_size, seq_len=512):
    while True:
        with xopen(file_path, "rt") as f:
            cr = csv.reader(f, delimiter="\t")
            next(cr) # Skip example number row.
            text = []
            labels = []
            for line in cr:
                if len(text) == batch_size:
                    # Fun fact: the 2 inputs must be in a list, *not* a tuple. Why.
                    yield ([np.asarray(text), np.zeros_like(text)], np.asarray(labels))
                    text = []
                    labels = []
                text.append(np.asarray(json.loads(line[0]))[0:seq_len])
                labels.append(np.asarray(json.loads(line[1])))
            # Yield what is left as the last batch when file has been read to its end.
            yield ([np.asarray(text), np.zeros_like(text)], np.asarray(labels))

def argparser():
    arg_parse = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    arg_parse.add_argument("--test", help="Processed test file.", metavar="FILE", required=True)
    arg_parse.add_argument("--model", help="Keras model to load.", metavar="FILE", required=True)
    arg_parse.add_argument("--labels", help="File containing names of labels, use if you want predicted labels to be output in addition to scores.", metavar="FILE", default=None)
    arg_parse.add_argument("--label_mapping", help="Mapping from N labels to all labels. Use if only training on top N labels.", metavar="FILE", default=None)
    arg_parse.add_argument("--f1_threshold_start", help="Positive label prediction threshold range start.", metavar="FLOAT", default=0.1, type=float)
    arg_parse.add_argument("--f1_threshold_end", help="Positive label prediction threshold range end, exclusive.", metavar="FLOAT", default=1.0, type=float)
    arg_parse.add_argument("--f1_threshold_step", help="Positive label prediction threshold range step.", metavar="FLOAT", default=0.1, type=float)
    arg_parse.add_argument("--output_file", help="File to which save the prediction results, .gz file extension recommended for compression!", metavar="FILE", default="predict_output.txt.gz")
    arg_parse.add_argument("--output_labels_threshold", help="Positive label prediction threshold for output file.", metavar="FLOAT", default=0.5, type=float)
    arg_parse.add_argument("--seq_len", help="BERT's maximum sequence length.", metavar="INT", default=512, type=int)
    arg_parse.add_argument("--clip_value", help="Set scores lower than this to 0.", metavar="FLOAT", default=1e-4, type=float)
    arg_parse.add_argument("--gpus", help="Number of GPUs to use.", metavar="INT", default=1, type=int)
    arg_parse.add_argument("--eval_batch_size", help="Batch size for eval calls. Default value is the Keras default.", metavar="INT", default=32, type=int)
    return arg_parse.parse_args()


def main(args):

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    set_session(tf.Session(config=config))

    print("Loading model..")
    custom_objects = get_custom_objects()
    model = load_model(args.model, custom_objects=custom_objects)

    with open(args.labels) as f:
        label_names = json.load(f)
        label_names = np.asarray(label_names)

    if args.label_mapping is not None:
        with open() as f:
            label_mapping = json.load(f)

    labels_true = []
    with xopen(args.test, "rt") as f:
        cr = csv.reader(f, delimiter="\t")
        next(cr) # Skip example number row.
        for line in tqdm(cr, desc="Loading test data"):
            labels_true.append(json.loads(line[1]))
        labels_true = lil_matrix(labels_true)

    label_to_index = {k:v for v,k in enumerate(label_names)}

    # from collections import defaultdict
    # parent_dict = defaultdict(list)

    import networkx

    # Build graph for label propagation.
    graph = networkx.DiGraph()

    with xopen("/mnt/extra_raid/sukaew/CAFA4/data/CAFA3/ctrl_sequences/strong/parent_term2term.txt.gz") as f:
        cr = csv.reader(f, delimiter='\t')
        next(cr) # skip header
        for line in cr:
            try:
            #parent_dict[label_to_index[line[1]]].append(label_to_index[line[0]])
                graph.add_edge(label_to_index[line[0]], label_to_index[line[1]])
            except KeyError: # Label not found in training data.
                pass

    #print(label_names[parent_dict[label_to_index["GO:0032042"]]])

    print("Predicting..")
    labels_prob = model.predict_generator(data_generator(args.test, args.eval_batch_size, seq_len=args.seq_len), use_multiprocessing=True,
                                                    steps=ceil(get_example_count(args.test) / args.eval_batch_size), verbose=1)
    # This makes json.dumps go faster and increases compression.
    labels_prob[labels_prob<args.clip_value] = 0
    print(labels_prob.shape)    

    missing_labels = set()

    for threshold in np.arange(args.f1_threshold_start, args.f1_threshold_end, args.f1_threshold_step):
        print("Threshold:", threshold)
        labels_pred = np.zeros(labels_prob.shape, dtype='b')
        labels_pred[labels_prob>threshold] = 1
        for example_index in tqdm(range(labels_pred.shape[0])):
            parent_indices = set()
            for label_index in np.where(labels_pred[example_index] == 1)[0]:
                try:
                    for parent in networkx.ancestors(graph, label_index):
                        parent_indices.add(parent)
                except:
                    missing_labels.add(label_names[label_index])
                    #print(label_names[label_index], "not found in graph!")
                #print(parent_indices)
            #print(labels_pred[example_index, list(parent_indices)])
            labels_pred[example_index, list(parent_indices)] = 1
            #print(labels_pred[example_index, list(parent_indices)]); input()
        precision, recall, f1, _ = precision_recall_fscore_support(labels_true, labels_pred, average="micro")
        print("Precision:", precision)
        print("Recall:", recall)
        print("F1-score:", f1, "\n")
        
    print("Labels not found in parent_term2term:", missing_labels)

    pos_labels = []
    for example in labels_prob:
        example_labels = []
        for i, label_prob in enumerate(example):
            if label_prob > args.output_labels_threshold:
                #print("Label found!")
                example_labels.append(label_names[i])
        pos_labels.append(example_labels)

    with xopen(args.output_file, "wt") as f:
        cw_out = csv.writer(f)
        for prob, labels in tqdm(zip(labels_prob, pos_labels), desc="Writing " + args.output_file):
            cw_out.writerow( (json.dumps(prob.tolist()), json.dumps(labels)) )

if __name__ == "__main__":
    main(argparser())