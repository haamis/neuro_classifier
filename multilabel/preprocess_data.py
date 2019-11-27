import csv, gzip, json, os, pickle, sys
import numpy as np

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

from itertools import chain

from functools import partial

from collections import Counter

from multiprocessing import Pool, cpu_count

from tqdm import tqdm

from sklearn.preprocessing import MultiLabelBinarizer

from bert import tokenization


def argparser():
    arg_parse = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    arg_parse.add_argument("-i", "--input_files", help="TSV input files: train, dev and test.", nargs=3, metavar="FILES", required=True)
    arg_parse.add_argument("-t", "--task", help="Task name.", choices=["bioasq", "cafa", "eng", "mesh_xml"], required=True)
    arg_parse.add_argument("-v", "--vocab", help="BERT vocab.", metavar="FILE", required=True)
    arg_parse.add_argument("-l", "--labels_out", help="List of labels in order.", metavar="FILE", default="labels.json")
    arg_parse.add_argument("-f", "--full_labels", help="List of all labels in order. Use only with top_n_labels.", metavar="FILE", default="full_labels.json")
    arg_parse.add_argument("-m", "--label_mapping", help="Mapping of labels partial -> full if using top_n_labels", metavar="FILE", default="label_mapping.json")
    arg_parse.add_argument("-d", "--do_lower_case", help="Lowercase input text.", metavar="bool", default=False)
    arg_parse.add_argument("-s", "--seq_len", help="BERT's max sequence length.", metavar="INT", type=int, default=512)
    arg_parse.add_argument("-n", "--top_n_labels", help="Only use top N labels. 0 for disabled", metavar="INT", type=int, default=0)
    arg_parse.add_argument("-p", "--processes", help="Number of parallel processes, uses all cores by default.", metavar="INT", type=int, default=cpu_count())
    return arg_parse.parse_args()

def bioasq_reader(file):
    examples = []
    labels = []
    csv_reader = csv.reader(file, delimiter=",")
    for line in tqdm(csv_reader, desc="Reading file"):
        examples.append(line[1])
        labels.append(json.dumps(line[0]))
    return examples, labels

def cafa_reader(file):
    examples = []
    labels = []
    csv_reader = csv.reader(file, delimiter="\t")
    for line in tqdm(csv_reader, desc="Reading file"):
        examples.append(line[1])
        labels.append(line[0].split(','))
    return examples, labels

def eng_reader(file):
    csv.field_size_limit(sys.maxsize)
    examples = []
    labels = []
    csv_reader = csv.reader(file, delimiter="\t")
    for line in tqdm(csv_reader, desc="Reading file"):
        examples.append(line[1])
        labels.append(line[0].split(' '))
    return examples, labels

def mesh_xml_reader(csv_reader):
    examples = []
    labels = []
    csv_reader = csv.reader(file, delimiter="\t")
    for line in tqdm(csv_reader, desc="Reading file"):
        examples.append(line[0])
        labels.append(line[1].split(','))
    return examples, labels

input_readers = {
    "bioasq" : bioasq_reader,
    "cafa" : cafa_reader,
    "eng" : eng_reader,
    "mesh_xml" : mesh_xml_reader,
}

def tokenize(text, tokenizer, maxlen=512):
    text = text.split(" ")[:maxlen]
    text = " ".join(text)
    return ["[CLS]"] + tokenizer.tokenize(text)[0:maxlen-2] + ["[SEP]"]

def vectorize(text, vocab, maxlen=512):
    return [vocab[token] for token in text] + [0] * (maxlen - len(text))

def preprocess_data(args):

    # Create child processes before we load a bunch of data.
    with Pool(args.processes) as p:
        examples_list = []
        labels_list = []
        for input_file in args.input_files:
            if input_file[-3:] == ".gz":
                open_fn = gzip.open 
            else:
                open_fn = open

            with open_fn(input_file, 'rt') as f:

                examples, labels = input_readers[args.task](f)

                tokenizer = tokenization.FullTokenizer(args.vocab, do_lower_case=args.do_lower_case)

                print("Tokenizing..")
                examples = p.map(partial(tokenize, tokenizer=tokenizer), examples)
                print("Vectorizing..")
                examples = p.map(partial(vectorize, vocab=tokenizer.vocab), examples)

                #print("Token vectors shape:", examples.shape)
                examples_list.append(examples)
                labels_list.append(labels)

    # Child processes terminated here.

    if args.top_n_labels > 0:
        print("Processing all labels first for mapping..")
        mlb_full = MultiLabelBinarizer(sparse_output=False)
        mlb_full = mlb_full.fit(chain.from_iterable(labels_list))
        print("Filtering to top", args.top_n_labels, "labels..")
        counter = Counter(chain(*chain.from_iterable(labels_list)))
        top_n_keys = set([k for k,v in counter.most_common(args.top_n_labels)])
        labels_list = [[[label for label in example_labels if label in top_n_keys] for example_labels in part_labels] for part_labels in labels_list]

    print("Binarizing labels..")
    mlb = MultiLabelBinarizer(sparse_output=False)
    mlb = mlb.fit(chain.from_iterable(labels_list))
    labels_list = [mlb.transform(labels).tolist() for labels in labels_list]
    #print("Labels shape:", labels_list[0].shape)

    # Save list of partial -> full mapping if doing top N labels.
    if args.top_n_labels > 0:
        label_mapping = np.where(np.in1d(mlb_full.classes_, mlb.classes_))[0].tolist()
        #for label in mlb.classes_:
        #    label_mapping.append(np.where(mlb_full.classes_ == label)[0])
        with open(args.label_mapping, "wt") as f:
            json.dump(label_mapping, f)
        # Also save the full labels.
        with open(args.full_labels, "wt") as f:
            json.dump(list(mlb_full.classes_), f)

    # Save list of labels using json module but the resulting string evaluates with list() just fine.
    with open(args.labels_out, "wt") as f:
        json.dump(list(mlb.classes_), f)

    for i, input_file in enumerate(args.input_files):
        file_basename = os.path.splitext(input_file)[0]
        if input_file[-3:] == ".gz":
            # Split twice to get rid of two file extensions, e.g. ".tsv.gz"
            file_basename = os.path.splitext(file_basename)[0]
        if args.top_n_labels > 0:
            file_name = file_basename + "-top-" + str(args.top_n_labels) + "-processed.gz"
        else:
            file_name = file_basename + "-processed.gz"
        with gzip.open(file_name, "wt") as f:
            print("Writing ", file_name)
            cw = csv.writer(f, delimiter="\t")
            # Write number of examples as the first row, useful for the finetuning.
            cw.writerow([len(examples_list[i])])
            for example, label in zip(examples_list[i], labels_list[i]):
                cw.writerow( (json.dumps(example), json.dumps(label)) )

if __name__ == '__main__':
    args = argparser()
    preprocess_data(args)
