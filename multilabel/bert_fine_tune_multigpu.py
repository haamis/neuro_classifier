import pickle, sys, json, csv, gzip
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
tf.logging.set_verbosity(tf.logging.ERROR)
import keras.backend as K

import numpy as np

from math import ceil

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

from scipy.sparse import lil_matrix

from keras.callbacks import Callback, EarlyStopping, ModelCheckpoint
from keras.layers import Dense, Flatten, Lambda, Dropout
from keras.models import Model
from keras.utils import multi_gpu_model

from keras_bert.loader import load_trained_model_from_checkpoint
from keras_bert import AdamWarmup, calc_train_steps

from sklearn.metrics import precision_recall_fscore_support

def argparser():
    arg_parse = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    arg_parse.add_argument("--train", help="Processed train file.", metavar="FILE", required=True)
    arg_parse.add_argument("--dev", help="Processed dev file.", metavar="FILE", required=True)
    arg_parse.add_argument("--init_checkpoint", help="BERT tensorflow model with path ending in .ckpt", metavar="PATH", required=True)
    arg_parse.add_argument("--bert_config", help="BERT config file.", metavar="FILE", required=True)
    arg_parse.add_argument("--batch_size", help="Batch size to use for finetuning.", metavar="INT", type=int, required=True)
    arg_parse.add_argument("--lr", "--learning_rate", help="Peak learning rate.", metavar="FLOAT", type=float, required=True)
    arg_parse.add_argument("--epochs", help="Max amount of epochs to run.", metavar="INT", type=int, required=True)
    arg_parse.add_argument("--output_file", help="Path to which save the finetuned model. Checkpoints will have the format `<output_file>.checkpoint-<epoch>`.", metavar="PATH", default="model.h5")
    arg_parse.add_argument("--seq_len", help="BERT's maximum sequence length.", metavar="INT", default=512, type=int)
    arg_parse.add_argument("--do_lower_case", help="Lowercase input text.", metavar="BOOL", default=False, type=bool)
    arg_parse.add_argument("--gpus", help="Number of GPUs to use.", metavar="INT", default=1, type=int)
    arg_parse.add_argument("--patience", help="Patience of early stopping. Early stopping disabled if -1.", metavar="INT", default=-1, type=int)
    arg_parse.add_argument("--threshold_start", help="Positive label prediction threshold range start.", metavar="FLOAT", default=0.1, type=float)
    arg_parse.add_argument("--threshold_end", help="Positive label prediction threshold range end, exclusive.", metavar="FLOAT", default=1.0, type=float)
    arg_parse.add_argument("--threshold_step", help="Positive label prediction threshold range step.", metavar="FLOAT", default=0.1, type=float)
    arg_parse.add_argument("--eval_batch_size", help="Batch size for eval calls. Default value is the Keras default.", metavar="INT", default=32, type=int)
    return arg_parse.parse_args()

# Read example count from the first row a preprocessed file.
def get_example_count(file_path):
    with gzip.open(file_path, "rt") as f:
        cr = csv.reader(f, delimiter="\t")
        return int(next(cr)[0])

# Read the first example from a file and return the length of the label list.
def get_label_dim(file_path):
    with gzip.open(file_path, "rt") as f:
        cr = csv.reader(f, delimiter="\t")
        next(cr) # Skip example number row.
        return len(json.loads(next(cr)[1]))

def data_generator(file_path, batch_size):
    while True:
        with gzip.open(file_path, "rt") as f:
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
                text.append(np.asarray(json.loads(line[0]))[0:args.seq_len])
                labels.append(np.asarray(json.loads(line[1])))
            # Yield what is left as the last batch when file has been read to its end.
            yield ([np.asarray(text), np.zeros_like(text)], np.asarray(labels))

class Metrics(Callback):

    def __init__(self):
        print("Metrics init, reading dev labels..")
        self.labels_dev = []
        with gzip.open(args.dev, "rt") as f:
            cr = csv.reader(f, delimiter="\t")
            next(cr) # Skip example number row.
            for line in cr:
                self.labels_dev.append(json.loads(line[1]))
            self.labels_dev = lil_matrix(self.labels_dev)
            print("Dev labels shape:", self.labels_dev.shape)

    def on_epoch_end(self, epoch, logs=None):
        print("Predicting probabilities..")
        labels_prob = self.model.predict_generator(data_generator(args.dev, args.eval_batch_size), use_multiprocessing=True,
                                                    steps=ceil( get_example_count(args.dev) / args.eval_batch_size))
        print("Probabilities to labels..")
        for threshold in np.arange(args.threshold_start, args.threshold_end, args.threshold_step):
            labels_pred = lil_matrix(labels_prob.shape, dtype='b')
            print("Threshold:", threshold)
            labels_pred[labels_prob>threshold] = 1
            precision, recall, f1, _ = precision_recall_fscore_support(self.labels_dev, labels_pred, average="micro")
            print("Precision:", precision)
            print("Recall:", recall)
            print("F1-score:", f1, "\n")


def build_model(args):

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    #config.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1
    K.set_session(tf.Session(config=config))

    print("Building model..")
    bert = load_trained_model_from_checkpoint(args.bert_config, args.init_checkpoint,
                                                training=False, trainable=True,
                                                seq_len=args.seq_len)

    slice_layer = Lambda(lambda x: K.slice(x, [0, 0, 0], [-1, 1, -1]))(bert.layers[-1].output)

    flatten_layer = Flatten()(slice_layer)

    dropout_layer = Dropout(0.1)(flatten_layer)

    output_layer = Dense(get_label_dim(args.train), activation='sigmoid')(dropout_layer)

    model = Model(bert.input, output_layer)

    if args.gpus > 1:
        template_model = model
        model = multi_gpu_model(template_model, gpus=args.gpus)

    callbacks = [Metrics()]

    if args.patience > -1:
        callbacks.append(EarlyStopping(patience=args.patience, verbose=1))

    callbacks.append(ModelCheckpoint(args.output_file + ".checkpoint-{epoch}"))

    total_steps, warmup_steps =  calc_train_steps(num_example=get_example_count(args.train),
                                                batch_size=args.batch_size, epochs=args.epochs,
                                                warmup_proportion=0.1)

    optimizer = AdamWarmup(total_steps, warmup_steps, lr=args.lr)

    model.compile(loss="binary_crossentropy", optimizer=optimizer, metrics=["accuracy"])
    
    print(model.summary(line_length=118))
    print("Number of GPUs in use:", args.gpus)
    print("Batch size:", args.batch_size)
    print("Learning rate:", args.lr)


    # ceil( get_example_count(args.train) / args.batch_size )
    model.fit_generator(data_generator(args.train, args.batch_size),
                        steps_per_epoch=ceil( get_example_count(args.train) / args.batch_size ), 
                        use_multiprocessing=True, epochs=args.epochs, callbacks=callbacks,
                        validation_data=data_generator(args.dev, args.eval_batch_size),
                        validation_steps=ceil( get_example_count(args.dev) / args.eval_batch_size ))
    
    if args.gpus > 1:
        template_model.save(args.output_file)
    else:
        model.save(args.output_file)

if __name__ == "__main__":

    args = argparser()
    build_model(args)
