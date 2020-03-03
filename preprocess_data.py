import csv, gzip, os, sys
csv.field_size_limit(sys.maxsize)

try:
    import ujson as json
except ImportError:
    import json

import numpy as np

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from itertools import chain
from functools import partial
from collections import Counter
from multiprocessing import Pool, cpu_count

from scipy.sparse import csr_matrix
from sklearn.preprocessing import MultiLabelBinarizer
from tqdm import tqdm
from xopen import xopen
from tokenizers import BertWordPieceTokenizer

from multiprocessing_generator import ParallelGenerator

def argparser():
    arg_parse = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    arg_parse.add_argument("-i", "--input_files", help="Input files: train, dev and test.", nargs='+', metavar="FILES", required=True)
    arg_parse.add_argument("-t", "--task", help="Task name.", choices=input_readers.keys(), required=True)
    arg_parse.add_argument("-v", "--vocab", help="BERT vocab.", metavar="FILE", required=True)
    arg_parse.add_argument("-l", "--labels_out", help="Path which to output labels file.", metavar="FILE", default="labels.json")
    arg_parse.add_argument("-f", "--full_labels", help="Path which to output list of all labels in order. Use only with top_n_labels.", metavar="FILE", default="full_labels.json")
    arg_parse.add_argument("-m", "--label_mapping", help="Path which to output mapping of labels partial -> full if using top_n_labels", metavar="FILE", default="label_mapping.json")
    arg_parse.add_argument("-d", "--do_lower_case", help="Lowercase input text.", metavar="bool", type=bool, default=False)
    arg_parse.add_argument("-s", "--seq_len", help="BERT's max sequence length.", metavar="INT", type=int, default=512)
    arg_parse.add_argument("-n", "--top_n_labels", help="Only use top N labels. 0 for disabled", metavar="INT", type=int, default=0)
    arg_parse.add_argument("-p", "--processes", help="Number of parallel processes, uses all cores by default.", metavar="INT", type=int, default=cpu_count())
    return arg_parse.parse_args()

def bioasq_reader(file):
    for line in tqdm(file, desc="Reading file"):
        line = json.loads(line)
        # Check title field for nulls and concat all the text fields.
        example = " ".join([ (line["title"] or ""), line["journal"], line["abstractText"] ])
        labels = line["meshMajor"]
        yield example, labels

def cafa_reader(file):
    csv_reader = csv.reader(file, delimiter="\t")
    for line in tqdm(csv_reader, desc="Reading file"):
        example = line[1]
        labels = line[0].split(',')
        yield example, labels

def eng_reader(file):
    csv_reader = csv.reader(file, delimiter="\t")
    for line in tqdm(csv_reader, desc="Reading file"):
        example = line[1]
        labels = line[0].split(' ')
        yield example, labels

input_readers = {
    "bioasq" : bioasq_reader,
    "cafa" : cafa_reader,
    "eng" : eng_reader,
}

def generate_out_filename(input_file, args):
    file_basename = os.path.splitext(input_file)[0]
    if input_file[-3:] == ".gz":
        # Split another time to get rid of two file extensions, e.g. ".tsv.gz"
        file_basename = os.path.splitext(file_basename)[0]
    if args.do_lower_case:
        file_basename = "-".join([file_basename, "uncased"])
    if args.top_n_labels > 0:
        file_basename = "-".join([file_basename, "-top-" + str(args.top_n_labels)])
    return file_basename + "-processed.jsonl.gz"

def preprocess_data(args):

    label_counter = Counter([])
    examples_per_file = Counter()

    print("Reading all files for labels.")
    for input_file in args.input_files:
        with xopen(input_file, "rt") as f:
            for example, labels in input_readers[args.task](f):
                examples_per_file[input_file] += 1
                label_counter.update(labels)

    if args.top_n_labels > 0:
        mlb_full = MultiLabelBinarizer(sparse_output=True)
        mlb_full = mlb_full.fit(label_counter.keys())
        label_counter = dict(label_counter.most_common(args.top_n_labels))

    mlb = MultiLabelBinarizer(sparse_output=True)
    # Passing a list in a list because that's what the function wants.
    mlb = mlb.fit([[pair for pair in label_counter]])

    # Save list of partial -> full mapping if doing top N labels.
    if args.top_n_labels > 0:

        label_mapping = np.where(np.in1d(mlb_full.classes_, mlb.classes_))[0].tolist()

        with xopen(args.label_mapping, "wt") as f:
            f.write(json.dumps(label_mapping))

        # Also save the full labels.
        with xopen(args.full_labels, "wt") as f:
            f.write(json.dumps(list(mlb_full.classes_)))

    # Save list of labels.
    with xopen(args.labels_out, "wt") as f:
        f.write(json.dumps(list(mlb.classes_)))

    # Set parallel tokenization thread count.
    os.environ["RAYON_NUM_THREADS"] = str(args.processes)

    from tokenizers import Tokenizer, decoders, trainers
    from tokenizers.models import WordPiece
    from tokenizers.normalizers import BertNormalizer
    from tokenizers.pre_tokenizers import BertPreTokenizer
    from tokenizers.processors import BertProcessing

    if args.task == 'cafa':
        # Define our custom tokenizer.
        # It is exactly the same as the default BERT tokenizer, except for max_input_chars_per_word
        # being 20000 instead of 100. This tokenizer is very slow on the long protein sequences.
        tokenizer = WordPiece.from_files(args.vocab, unk_token="[UNK]", max_input_chars_per_word=20000)
        tokenizer = Tokenizer(tokenizer)
        tokenizer.add_special_tokens(["[UNK]", "[SEP]", "[CLS]"])
        tokenizer.normalizer = BertNormalizer(lowercase=args.do_lower_case)
        tokenizer.pre_tokenizer = BertPreTokenizer()
        tokenizer.post_processor = BertProcessing( ("[SEP]", tokenizer.token_to_id("[SEP]")), ("[CLS]", tokenizer.token_to_id("[CLS]")) )
        tokenizer.decoder = decoders.WordPiece(prefix='##')
    else:
        tokenizer = BertWordPieceTokenizer(args.vocab, lowercase=args.do_lower_case)

    tokenizer.enable_padding(max_length=args.seq_len)
    tokenizer.enable_truncation(max_length=args.seq_len)

    for input_file in args.input_files:
        with xopen(input_file, 'rt') as in_f:
            
            file_name = generate_out_filename(input_file, args)
            
            with xopen(file_name, "wt") as out_f:
                print("Processing to: ", file_name)
                
                # Write the shape as the first row, useful for the finetuning.
                out_f.write(json.dumps( (examples_per_file[input_file], len(label_counter)) ) + '\n')

                batch_size = min(examples_per_file[input_file], args.processes*100)
                example_batch = []
                labels_batch = []

                with ParallelGenerator(input_readers[args.task](in_f), max_lookahead=batch_size) as g:
                    for example, labels in g:

                        example_batch.append(example)
                        labels_batch.append(labels)

                        if len(example_batch) == batch_size:
                            example_batch = tokenizer.encode_batch(example_batch)
                            labels_batch = mlb.transform(labels_batch)
                        
                            for example, labels in zip(example_batch, labels_batch):
                                # Convert sparse arrays to python lists for json dumping.
                                # print(labels);input()
                                labels = labels.nonzero()[1].tolist()
                                out_f.write(json.dumps( [example.ids, labels] ) + '\n')

                            example_batch = []
                            labels_batch = []

                    # Write out whatever is left in the last smaller batch.
                    example_batch = tokenizer.encode_batch(example_batch)
                    labels_batch = mlb.transform(labels_batch)
                
                    for example, labels in zip(example_batch, labels_batch):
                        # Convert sparse arrays to python lists for json dumping.
                        # print(labels);input()
                        labels = labels.nonzero()[1].tolist()
                        out_f.write(json.dumps( [example.ids, labels] ) + '\n')
                        

 
if __name__ == '__main__':
    args = argparser()
    preprocess_data(args)
