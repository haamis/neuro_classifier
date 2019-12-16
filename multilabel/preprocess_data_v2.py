import csv, gzip, json, os, pickle, sys
import numpy as np
import orjson as json

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from itertools import chain
from functools import partial
from collections import Counter
from multiprocessing import Pool, cpu_count

from scipy.sparse import csr_matrix
from sklearn.preprocessing import MultiLabelBinarizer
from tqdm import tqdm
from xopen import xopen
from bert import tokenization

from multiprocessing_generator import ParallelGenerator

def argparser():
    arg_parse = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    arg_parse.add_argument("-i", "--input_files", help="TSV input files: train, dev and test.", nargs='+', metavar="FILES", required=True)
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

# Yields (example, labels).
def bioasq_reader(file, yield_size=1):
    #examples = []
    #labels = []
    csv_reader = csv.reader(file, delimiter="\t")
    for line in tqdm(csv_reader, desc="Reading file"):
        example = line[1]
        # labels = json.loads(line[0])
        labels = line[0].split(';')
        yield example, labels

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
    return csr_matrix([vocab[token] for token in text] + [0] * (maxlen - len(text)), dtype="int32")

def worker_function(queue):
    tokenizer = tokenization.FullTokenizer(args.vocab, do_lower_case=args.do_lower_case)
    while True:
        item = queue.get()
        item[0] = tokenize(item[0], tokenizer=tokenizer, maxlen=args.seq_len)
        item[0] = vectorize(item[0], vocab=tokenizer.vocab, maxlen=args.seq_len)
        item[1] = mlb.transform(item[1])

def sparse_to_json(matrix):
    return json.dumps(matrix[0].todense().tolist(), matrix[1].todense().tolist()) + b'\n'

def preprocess_data(args):

    label_counter = Counter([])
    examples_per_file = Counter()

    print("Reading all files for labels.")
    for input_file in args.input_files:
        with xopen(input_file, "rt") as f:
            #_, labels = input_readers[args.task](f)
            for example, labels in input_readers[args.task](f):
                examples_per_file[input_file] += 1
                label_counter.update(labels)
            #c.update(chain.from_iterable(input_readers[args.task](f)))

    print(len(label_counter.most_common()))

    if args.top_n_labels > 0:
        mlb_full = MultiLabelBinarizer(sparse_output=True)
        #print(label_counter.keys())
        mlb_full = mlb_full.fit(label_counter.keys())
        #counter = Counter(chain.from_iterable(chain.from_iterable(labels_list)))
        #top_n_keys = set([k for k,v in c.most_common(args.top_n_labels)])
        label_counter = dict(label_counter.most_common(args.top_n_labels))
        #print(label_counter)
        #print(label_counter)
        #labels_list = [[[label for label in example_labels if label in top_n_keys] for example_labels in part_labels] for part_labels in labels_list]

    mlb = MultiLabelBinarizer(sparse_output=True)
    # Passing a list in a list because that's what the function wants.
    mlb = mlb.fit([[pair for pair in label_counter]])
    print(mlb.classes_)
    #input()

    # Save list of partial -> full mapping if doing top N labels.
    if args.top_n_labels > 0:

        # Fancy way:
        label_mapping = np.where(np.in1d(mlb_full.classes_, mlb.classes_))[0].tolist()
        # Simpler way:
        # for label in mlb.classes_:
        #     label_mapping.append(np.where(mlb_full.classes_ == label)[0])

        with open(args.label_mapping, "wb") as f:
            f.write(json.dumps(label_mapping))

        # Also save the full labels.
        with open(args.full_labels, "wb") as f:
            f.write(json.dumps(list(mlb_full.classes_)))

    # Save list of labels.
    with open(args.labels_out, "wb") as f:
        f.write(json.dumps(list(mlb.classes_)))
    
    # We want to have 1 core for the main thread.
    num_processes = args.processes - 1
    # Just in case.
    if num_processes < 1:
        num_processes = 1

    # Create child processes before we load a bunch of data.
    with Pool(num_processes) as p:
        tokenizer = tokenization.FullTokenizer(args.vocab, do_lower_case=args.do_lower_case)
        for input_file in args.input_files:
            with xopen(input_file, 'rt') as in_f:
                
                # Figure out filename.
                file_basename = os.path.splitext(input_file)[0]
                if input_file[-3:] == ".gz":
                    # Split another time to get rid of two file extensions, e.g. ".tsv.gz"
                    file_basename = os.path.splitext(file_basename)[0]
                if args.top_n_labels > 0:
                    file_name = file_basename + "-top-" + str(args.top_n_labels) + "-processed.gz"
                else:
                    file_name = file_basename + "-processed.gz"
                
                with xopen(file_name, "wt") as out_f:
                    print("Writing ", file_name)
                    cw = csv.writer(out_f, delimiter="\t")

                    # Write number of examples as the first row, useful for the finetuning.
                    cw.writerow([examples_per_file[input_file]])

                    example_batch = []
                    labels_batch = []
                    with ParallelGenerator(input_readers[args.task](in_f), max_lookahead=num_processes*1000) as g:
                        for example, labels in g:
                            example_batch.append(example)
                            labels_batch.append(labels)
                            if len(example_batch) == num_processes*1000:
                                #print(labels_batch)
                                example_batch = p.map(partial(tokenize, tokenizer=tokenizer, maxlen=args.seq_len), example_batch)
                                example_batch = p.map(partial(vectorize, vocab=tokenizer.vocab, maxlen=args.seq_len), example_batch)
                                labels_batch = p.map(mlb.transform, [labels_batch])
                                #print(labels_out)
                                
                                # Convert sparse arrays to python lists for json dumping.
                                cw.writerows(zip(example_batch, labels_batch))
                                example_batch = []
                                labels_batch = []

 
if __name__ == '__main__':
    args = argparser()
    preprocess_data(args)
