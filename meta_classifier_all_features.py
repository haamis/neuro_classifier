import csv, os, sys
csv.field_size_limit(sys.maxsize)

try:
    import ujson as json
except ImportError:
    import json

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
tf.logging.set_verbosity(tf.logging.ERROR)

import keras
import keras.backend as K
import numpy as np

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from math import ceil

from xopen import xopen

from tqdm import tqdm

from scipy.sparse import lil_matrix
from sklearn.metrics import precision_recall_fscore_support

from keras.callbacks import Callback, EarlyStopping, ModelCheckpoint
from keras.layers import Activation, Average, Maximum, Concatenate, Conv1D, Dense, Dropout, Flatten, GlobalAveragePooling1D, GlobalMaxPooling1D, Lambda, Reshape, Permute, Input, TimeDistributed
from keras.models import Model
from keras.optimizers import Adam
from keras.utils import multi_gpu_model

from keras.models import load_model
from keras_bert import get_custom_objects
from keras_bert import AdamWarmup, calc_train_steps
from keras_self_attention import SeqSelfAttention


def argparser():
    arg_parse = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    arg_parse.add_argument("--model", help="Finetuned model.", metavar="FILE", required=True)
    arg_parse.add_argument("--train", help="Processed train file.", metavar="FILE", required=True)
    arg_parse.add_argument("--dev", help="Processed dev file.", metavar="FILE", required=True)
    arg_parse.add_argument("--features_train", help="Features file", metavar="FILE", nargs='+', required=True)
    arg_parse.add_argument("--features_dev", help="Features file", metavar="FILE", nargs='+', required=True)
    arg_parse.add_argument("--init_checkpoint", help="BERT tensorflow model with path ending in .ckpt", metavar="PATH", required=True)
    arg_parse.add_argument("--bert_config", help="BERT config file.", metavar="FILE", required=True)
    arg_parse.add_argument("--batch_size", help="Batch size to use for finetuning.", metavar="INT", type=int, required=True)
    arg_parse.add_argument("--lr", "--learning_rate", help="Peak learning rate.", metavar="FLOAT", type=float, required=True)
    arg_parse.add_argument("--epochs", help="Max amount of epochs to run.", metavar="INT", type=int, required=True)
    arg_parse.add_argument("--dev_all", help="Processed dev file with all labels. Use if only training on top N labels.", metavar="FILE", default=None)
    arg_parse.add_argument("--label_mapping", help="Mapping from N labels to all labels. Use if only training on top N labels.", metavar="FILE", default=None)
    arg_parse.add_argument("--output_file", help="Path to which save the finetuned model. Checkpoints will have the format `<output_file>.checkpoint-<epoch>`.", metavar="PATH", default="model.h5")
    arg_parse.add_argument("--seq_len", help="BERT's maximum sequence length.", metavar="INT", default=512, type=int)
    arg_parse.add_argument("--dropout", help="Dropout rate between BERT and the decision layer.", metavar="FLOAT", default=0.1, type=float)
    arg_parse.add_argument("--gpus", help="Number of GPUs to use.", metavar="INT", default=1, type=int)
    arg_parse.add_argument("--patience", help="Patience of early stopping. Early stopping disabled if -1.", metavar="INT", default=-1, type=int)
    arg_parse.add_argument("--checkpoint_interval", help="Interval between checkpoints. 1 for every epoch, 0 to disable.", metavar="INT", default=0, type=int)
    arg_parse.add_argument("--threshold_start", help="Positive label prediction threshold range start.", metavar="FLOAT", default=0.1, type=float)
    arg_parse.add_argument("--threshold_end", help="Positive label prediction threshold range end, exclusive.", metavar="FLOAT", default=1.0, type=float)
    arg_parse.add_argument("--threshold_step", help="Positive label prediction threshold range step.", metavar="FLOAT", default=0.1, type=float)
    arg_parse.add_argument("--eval_batch_size", help="Batch size for eval calls. Default value is the Keras default.", metavar="INT", default=32, type=int)
    return arg_parse.parse_args()

# Read example count from the first row a preprocessed file.
def get_example_count(file_path):
    with xopen(file_path, "rt") as f:
        cr = csv.reader(f, delimiter="\t")
        return int(next(cr)[0])

# Read the first example from a file and return the length of the label list.
def get_label_dim(file_path):
    with xopen(file_path, "rt") as f:
        cr = csv.reader(f, delimiter="\t")
        next(cr) # Skip example number row.
        return len(json.loads(next(cr)[1]))

def data_generator(file_path, batch_size, seq_len=512, features=None):
    while True:
        features_f = [xopen(feature, "rt") for feature in features]
        with xopen(file_path, "rt") as f:
            cr = csv.reader(f, delimiter="\t")
            cr_features = [csv.reader(feature_f, delimiter="\t") for feature_f in features_f]
            next(cr) # Skip example number row.
            text = []
            feature_labels = [[] for feature in features]
            labels = []
            for line, *line_features in zip(cr, *cr_features):
                #print(line_features[:]); input()
                if len(text) == batch_size:
                    # Fun fact: the 2 inputs must be in a list, *not* a tuple. Why.
                    yield ([np.asarray(text), np.zeros_like(text), *[np.asarray(feature_label) for feature_label in feature_labels] ], np.asarray(labels))
                    text = []
                    feature_labels = [[] for feature in features]
                    labels = []
                text.append(np.asarray(json.loads(line[0]))[0:seq_len])
                for i, line_feature in enumerate(line_features):
                    feature_labels[i].append(np.asarray(json.loads(line_feature[0])))
                labels.append(np.asarray(json.loads(line[1])))
            # Yield what is left as the last batch when file has been read to its end.
            yield ([np.asarray(text), np.zeros_like(text), *[np.asarray(feature_label) for feature_label in feature_labels] ], np.asarray(labels))

class Metrics(Callback):

    def __init__(self):

        self.best_f1 = 0
        self.best_f1_epoch = 0
        self.best_f1_threshold = 0

        self.all_labels = []
        if args.label_mapping is not None:
            file_name = args.dev_all
        else:
            file_name = args.dev
        with xopen(file_name, "rt") as f:
            cr = csv.reader(f, delimiter="\t")
            next(cr) # Skip example number row.
            for line in tqdm(cr, desc="Reading dev labels"):
                self.all_labels.append(json.loads(line[1]))
            self.all_labels = lil_matrix(self.all_labels, dtype='b')
            print("Dev labels shape:", self.all_labels.shape)
        if args.dev_all is not None:
            with xopen(args.label_mapping) as f:
                self.labels_mapping = json.loads(f.read())

    def on_epoch_end(self, epoch, logs=None):
        print(self.model.get_layer("time_distributed_1").get_weights())
        if epoch > -1:
            print("Predicting probabilities..")
            labels_prob = self.model.predict_generator(data_generator(args.dev, args.eval_batch_size, seq_len=args.seq_len, features=args.features_dev), use_multiprocessing=True,
                                                        steps=ceil(get_example_count(args.dev) / args.eval_batch_size), verbose=1)
            print("Probabilities to labels..")
            if args.label_mapping is not None:
                full_labels_prob = np.zeros(self.all_labels.shape)
                for i, probs in enumerate(labels_prob):
                    np.put(full_labels_prob[i], self.labels_mapping, probs)

                labels_prob = full_labels_prob
            
            for threshold in np.arange(args.threshold_start, args.threshold_end, args.threshold_step):
                print("Threshold:", threshold)
                labels_pred = lil_matrix(labels_prob.shape, dtype='b')
                labels_pred[labels_prob>=threshold] = 1
                precision, recall, f1, _ = precision_recall_fscore_support(self.all_labels, labels_pred, average="micro")
                if f1 > self.best_f1:
                    self.best_f1 = f1
                    self.best_f1_epoch = epoch
                    self.best_f1_threshold = threshold
                print("Precision:", precision)
                print("Recall:", recall)
                print("F1-score:", f1, "\n")

            print("Current F_max:", self.best_f1, "epoch", self.best_f1_epoch+1, "threshold", self.best_f1_threshold, '\n')


def loss_fn(y_true, y_pred):
    return keras.losses.binary_crossentropy(y_true, y_pred) + keras.losses.categorical_hinge(y_true, y_pred)

def build_model(args):

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    #config.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1
    K.set_session(tf.Session(config=config))
    
    print("Loading model..")
    custom_objects = get_custom_objects()
    bert_model = load_model(args.model, custom_objects=custom_objects)
    
    for layer in bert_model.layers:
        layer.trainable = False
    
    input_features = [Input(shape=(get_label_dim(args.train),)) for _ in args.features_train]

    stacked = Lambda(lambda x: K.stack(x, axis=1))([bert_model.output, *input_features])

    stacked = Permute((2, 1), name="stack_permute")(stacked)

    output_layer = TimeDistributed(Dense(1, activation="tanh", name="decision"))(stacked)
    output_layer = Flatten(name="time_distributed_flatten")(output_layer)
    output_layer = Activation("softmax")(output_layer)

    # The bert model has multiple inputs, so unpack those.
    model = Model([*bert_model.input, *input_features], output_layer)

    if args.gpus > 1:
        template_model = model
        model = multi_gpu_model(template_model, gpus=args.gpus)

    callbacks = [Metrics()]

    if args.patience > -1:
        callbacks.append(EarlyStopping(patience=args.patience, verbose=1))

    if args.checkpoint_interval > 0:
        callbacks.append(ModelCheckpoint(args.output_file + ".checkpoint-{epoch}",  period=args.checkpoint_interval))

    total_steps, warmup_steps =  calc_train_steps(num_example=get_example_count(args.train),
                                                batch_size=args.batch_size, epochs=args.epochs,
                                                warmup_proportion=0.01)

    optimizer = AdamWarmup(total_steps, warmup_steps, lr=args.lr)

    model.compile(loss=["categorical_crossentropy"], optimizer=optimizer, metrics=[])

    print(model.summary(line_length=118))
    print("Number of GPUs in use:", args.gpus)
    print("Batch size:", args.batch_size)
    print("Learning rate:", args.lr)
    print("Dropout:", args.dropout)

    model.fit_generator(data_generator(args.train, args.batch_size, seq_len=args.seq_len, features=args.features_train),
                        steps_per_epoch=ceil( get_example_count(args.train) / args.batch_size ),
                        use_multiprocessing=True, epochs=args.epochs, callbacks=callbacks,
                        validation_data=data_generator(args.dev, args.eval_batch_size, seq_len=args.seq_len, features=args.features_dev),
                        validation_steps=ceil( get_example_count(args.dev) / args.eval_batch_size ))
                        

    print("Saving model:", args.output_file)
    if args.gpus > 1:
        template_model.save(args.output_file)
    else:
        model.save(args.output_file)

if __name__ == "__main__":

    args = argparser()
    build_model(args)
