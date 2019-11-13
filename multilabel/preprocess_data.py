import csv, gzip, json, os, pickle
import numpy as np

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

from itertools import chain

from functools import partial

from multiprocessing import Pool, cpu_count

from tqdm import tqdm

from sklearn.preprocessing import MultiLabelBinarizer

from bert import tokenization


def argparser():
    arg_parse = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    arg_parse.add_argument("-i", "--input_files", help="TSV input files: train, dev and test.", nargs=3, metavar="FILES", required=True)
    arg_parse.add_argument("-t", "--task", help="Task name.", choices=["mesh", "cafa", "eng"], required=True)
    arg_parse.add_argument("-v", "--vocab", help="BERT vocab.", metavar="FILE", required=True)
    arg_parse.add_argument("-l", "--labels_out", help="List of labels in the order of the binarizer.", metavar="FILE", default="labels.txt")
    arg_parse.add_argument("-d", "--do_lower_case", help="Lowercase input text.", metavar="bool", default=False)
    arg_parse.add_argument("-s", "--seq_len", help="BERT's max sequence length.", metavar="INT", default=512)
    arg_parse.add_argument("-p", "--processes", help="Number of parallel processes, uses all cores by default.", metavar="INT", default=cpu_count())
    return arg_parse.parse_args()

def bioasq_reader(csv_reader):
    examples = []
    labels = []
    for line in tqdm(csv_reader, desc="Reading file"):
        examples.append(line[1])
        labels.append(json.dumps(line[0]))
    return examples, labels

def cafa_reader(csv_reader):
    examples = []
    labels = []
    for line in tqdm(csv_reader, desc="Reading file"):
        examples.append(line[1])
        labels.append(line[0].split(','))
    return examples, labels

def eng_reader(csv_reader):
    examples = []
    labels = []
    for line in tqdm(csv_reader, desc="Reading file"):
        examples.append(line[1])
        labels.append(line[0].split(' '))
    return examples, labels    

def mesh_xml_reader(csv_reader):
    examples = []
    labels = []
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
def tokenize(abstract, tokenizer, maxlen=512):
    return ["[CLS]"] + tokenizer.tokenize(abstract)[0:maxlen-2] + ["[SEP]"]

def vectorize(abstract, vocab, maxlen=512):
    return [vocab[token] for token in abstract] + [0] * (maxlen - len(abstract))

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
                
                data = csv.reader(f, delimiter="\t")

                examples, labels = input_readers[args.task](data)

                tokenizer = tokenization.FullTokenizer(args.vocab, do_lower_case=args.do_lower_case)

                print("Tokenizing..")
                examples = p.map(partial(tokenize, tokenizer=tokenizer), examples)
                print("Vectorizing..")
                examples = p.map(partial(vectorize, vocab=tokenizer.vocab), examples)

                # Int32 to save half the space over default int64.
                #examples = np.asarray(examples, dtype='int32')
        
                #print("Token vectors shape:", examples.shape)
                examples_list.append(examples)
                labels_list.append(labels)
    
    # Child processes terminated here.
    
    print("Binarizing labels..")
    mlb = MultiLabelBinarizer(sparse_output=False)
    mlb = mlb.fit(chain.from_iterable(labels_list))
    labels_list = [mlb.transform(labels).tolist() for labels in labels_list]
    #print("Labels shape:", labels_list[0].shape)

    # Save list of labels using json module but the resulting string evaluates with list() just fine.
    with open(args.labels_out, "wt") as f:
        json.dump(list(mlb.classes_), f)

    for i, input_file in enumerate(args.input_files):
        file_basename = os.path.splitext(input_file)[0]
        if input_file[-3:] == ".gz":
            # Split twice to get rid of two file extensions, e.g. ".tsv.gz"
            file_basename = os.path.splitext(file_basename)[0]
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
