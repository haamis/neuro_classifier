import os, sys

try:
    import ujson as json
except ImportError:
    import json

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
# tf.logging.set_verbosity(tf.logging.ERROR)

import keras
import keras.backend as K
import numpy as np
import horovod.keras as hvd

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from math import ceil

from xopen import xopen

from tqdm import tqdm

from scipy.sparse import lil_matrix
from sklearn.metrics import precision_recall_fscore_support

from keras.callbacks import Callback, EarlyStopping, ModelCheckpoint
from keras.layers import Average, Maximum, Concatenate, Conv1D, Dense, Dropout, Flatten, GlobalAveragePooling1D, GlobalMaxPooling1D, Lambda, Reshape, Permute
from keras.models import Model, load_model
from keras.utils import multi_gpu_model

from keras_bert.loader import load_trained_model_from_checkpoint
from keras_bert import AdamWarmup, calc_train_steps, get_custom_objects
from keras_bert.activations import gelu
from keras_transformer import get_encoder_component


def argparser():
    arg_parse = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    arg_parse.add_argument("--load_model", help="Keras model to continue from", metavar="FILE")
    arg_parse.add_argument("--init_checkpoint", help="BERT tensorflow model with path ending in .ckpt", metavar="PATH")
    arg_parse.add_argument("--train", help="Processed train file.", metavar="FILE", required=True)
    arg_parse.add_argument("--dev", help="Processed dev file.", metavar="FILE", required=True)
    arg_parse.add_argument("--bert_config", help="BERT config file.", metavar="FILE", default=None)
    arg_parse.add_argument("--batch_size", help="Batch size to use for finetuning.", metavar="INT", type=int, required=True)
    arg_parse.add_argument("--epochs", help="Max amount of epochs to run.", metavar="INT", type=int, required=True)
    arg_parse.add_argument("--lr", "--learning_rate", help="Peak learning rate.", metavar="FLOAT", type=float, default=5e-5)
    arg_parse.add_argument("--dev_all", help="Processed dev file with all labels. Use if only training on top N labels.", metavar="FILE", default=None)
    arg_parse.add_argument("--label_mapping", help="Mapping from N labels to all labels. Use if only training on top N labels.", metavar="FILE", default=None)
    arg_parse.add_argument("--output_file", help="Path to which save the finetuned model. Checkpoints will have the format `<output_file>.checkpoint-<epoch>`.", metavar="PATH", default="model.h5")
    arg_parse.add_argument("--seq_len", help="BERT's maximum sequence length.", metavar="INT", default=512, type=int)
    # arg_parse.add_argument("--dropout", help="Dropout rate between BERT and the decision layer.", metavar="FLOAT", default=0.1, type=float)
    # arg_parse.add_argument("--gpus", help="Number of GPUs to use.", metavar="INT", default=1, type=int)
    arg_parse.add_argument("--patience", help="Patience of early stopping. Early stopping disabled if -1.", metavar="INT", default=-1, type=int)
    arg_parse.add_argument("--checkpoint_interval", help="Interval between checkpoints. 1 for every epoch, 0 to disable.", metavar="INT", default=0, type=int)
    arg_parse.add_argument("--threshold_start", help="Positive label prediction threshold range start.", metavar="FLOAT", default=0.1, type=float)
    arg_parse.add_argument("--threshold_end", help="Positive label prediction threshold range end, exclusive.", metavar="FLOAT", default=1.0, type=float)
    arg_parse.add_argument("--threshold_step", help="Positive label prediction threshold range step.", metavar="FLOAT", default=0.1, type=float)
    arg_parse.add_argument("--eval_batch_size", help="Batch size for eval calls. Default value is the Keras default.", metavar="INT", default=32, type=int)
    arg_parse.add_argument("--print_lr", help="Print current learning rate for each batch.", action='store_true')
    parsed = arg_parse.parse_args()
    if parsed.load_model and (parsed.init_checkpoint or parsed.bert_config):
        arg_parse.error("--load_model and (--init_checkpoint and/or --bert_config) are mutually exclusive.")
    return arg_parse.parse_args()

# Read example count from the first row of a preprocessed file.
def get_example_count(file_path):
    with xopen(file_path, "rt") as f:
        return json.loads(f.readline())[0]

# Read label dimension from the first row of a preprocessed file.
def get_label_dim(file_path):
    with xopen(file_path, "rt") as f:
        return json.loads(f.readline())[1]

def data_generator(file_path, batch_size, seq_len=512, predict=False):
    # Trick the code into thinking we're only running 1 process for prediction when running `Metrics`.
    if predict:
        size = 1
    else:
        size = hvd.size()
    total_batch_size = batch_size * size
    print(total_batch_size)
    rank = hvd.rank()
    print(rank)
    range_start = batch_size * rank
    range_end = range_start + batch_size
    print(range_start, range_end)
    while True:
        with xopen(file_path, "rt") as f:
            _, label_dim = json.loads(f.readline())
            text = []
            labels = []
            for line in f:
                if len(text) == total_batch_size:
                    text = text[range_start:range_end]
                    labels = labels[range_start:range_end]
                    print(text[0])
                    # Fun fact: the 2 inputs must be in a list, *not* a tuple. Why.
                    yield ([np.asarray(text), np.zeros_like(text)], np.asarray(labels))
                    text = []
                    labels = []
                line = json.loads(line)
                # First sublist is token ids.
                text.append(np.asarray(line[0])[0:seq_len])

                # Second sublist is positive label indices.
                label_line = np.zeros(label_dim, dtype='b')
                label_line[line[1]] = 1
                labels.append(label_line)
            # Yield what is left as the last batch when file has been read to its end.
            # Split the remaining examples, duplicating with `ceil()` if they don't split evenly.
            leftover_batch_start = ceil(len(text) / size) * rank
            leftover_batch_end = leftover_batch_start + ceil(len(text) / size)
            text = text[leftover_batch_start:leftover_batch_end]
            labels = labels[leftover_batch_start:leftover_batch_end]
            yield ([np.asarray(text), np.zeros_like(text)], np.asarray(labels))


class Metrics(Callback):

    def __init__(self, model):
        
        print("Initing metrics")
        self.best_f1 = 0
        self.best_f1_epoch = 0
        self.best_f1_threshold = 0

        if args.label_mapping is not None:
            file_name = args.dev_all
        else:
            file_name = args.dev
        with xopen(file_name, "rt") as f:
            example_count, label_dim = json.loads(f.readline())
            self.all_labels = lil_matrix((example_count, label_dim), dtype='b')
            for i, line in tqdm(enumerate(f), desc="Reading dev labels"):
                self.all_labels[i, json.loads(line)[1]] = 1
            print("Dev labels shape:", self.all_labels.shape)
        if args.dev_all is not None:
            with xopen(args.label_mapping) as f:
                self.labels_mapping = json.loads(f.read())

    def on_epoch_end(self, epoch, logs):
        print("Predicting probabilities..")
        labels_prob = self.model.predict_generator(data_generator(args.dev, args.eval_batch_size, seq_len=args.seq_len, predict=True), use_multiprocessing=True,
                                                    steps=ceil(get_example_count(args.dev) / args.eval_batch_size), verbose=1)
        if args.label_mapping is not None:
            full_labels_prob = np.zeros(self.all_labels.shape)
            for i, probs in enumerate(labels_prob):
               np.put(full_labels_prob[i], self.labels_mapping, probs)

            labels_prob = full_labels_prob
        
        # batch_size = 1024
        # eval_loss = 0
        # true_data = self.all_labels[0:batch_size].todense()
        # pred_data = labels_prob[0:batch_size]
        # y_true = K.variable(true_data)
        # y_pred = K.variable(pred_data)
        # loss=keras.losses.binary_crossentropy(y_true, y_pred)
        # for i in range(0, get_example_count(args.dev), batch_size):
        #     print("y_true assigned")
        #     true_data = self.all_labels[i:i+batch_size].todense()
        #     print("y_pred assigned")
        #     pred_data = labels_prob[i:i+batch_size]
        #     y_true = K.variable(true_data)
        #     y_pred = K.variable(pred_data)
        #     print("loss assigned")
        #     loss_eval = sum(loss)/batch_size
        #     print(loss_eval)
        #     eval_loss += loss_eval

        # print("eval_loss:", eval_loss)

        print("Probabilities to labels..")
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


class PrintLR(Callback):
    def __init__(self, model):
        print("Initing lrprint")
        t = K.cast(model.optimizer.iterations, K.floatx()) + 1
        self.lr = K.switch(
            t <= model.optimizer.warmup_steps,
            model.optimizer.lr * (t / model.optimizer.warmup_steps),
            model.optimizer.min_lr + (model.optimizer.lr - model.optimizer.min_lr) * (1.0 - K.minimum(t, model.optimizer.decay_steps) / model.optimizer.decay_steps),
        )
    
    def on_batch_end(self, batch, logs):
        print(" - lr:", K.eval(self.lr), end='')


def build_model(args):

    hvd.init()

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.gpu_options.visible_device_list = str(hvd.local_rank())
    K.set_session(tf.Session(config=config))

    if args.load_model:
        print("Loading previously saved model..")
        if args.bert_config:
            print("Warning: --bert_config ignored when loading previous Keras model.", file=sys.stderr)
        custom_objects = get_custom_objects()
        model = load_model(args.load_model, custom_objects=custom_objects)
    
    else:
        print("Building model..")
        bert = load_trained_model_from_checkpoint(args.bert_config, args.init_checkpoint,
                                                    training=False, trainable=True,
                                                    seq_len=args.seq_len)
        print("BERT loaded", hvd.rank())

        transformer_output = get_encoder_component(name="Encoder-13", input_layer=bert.layers[-1].output,
                                                head_num=12, hidden_dim=3072, feed_forward_activation=gelu)

        drop_mask = Lambda(lambda x: x, name="drop_mask")(bert.output)

        slice_CLS = Lambda(lambda x: K.slice(x, [0, 0, 0], [-1, 1, -1]), name="slice_CLS")(drop_mask)
        flatten_CLS = Flatten()(slice_CLS)

        # Needed to avoid a json serialization error when saving the model.
        last_position = args.seq_len-1
        slice_SEP = Lambda(lambda x: K.slice(x, [0, last_position, 0], [-1, 1, -1]), name="slice_SEP")(drop_mask)
        flatten_SEP = Flatten()(slice_SEP)

        permute_layer = Permute((2, 1))(drop_mask)
        permute_average = GlobalAveragePooling1D()(permute_layer)
        permute_maximum =  GlobalMaxPooling1D()(permute_layer)

        concat = Concatenate()([permute_average, permute_maximum, flatten_CLS, flatten_SEP])

        output_layer = Dense(get_label_dim(args.train), activation='sigmoid', name="label_out")(flatten_CLS)

        model = Model(bert.input, output_layer)
        print("model created", hvd.rank())
        
        total_steps, warmup_steps =  calc_train_steps(num_example=get_example_count(args.train),
                                                    batch_size=args.batch_size, epochs=args.epochs,
                                                    warmup_proportion=0.01)

        # optimizer = AdamWarmup(total_steps, warmup_steps, lr=args.lr)
        optimizer = keras.optimizers.Adam(lr=args.lr)
        optimizer = hvd.DistributedOptimizer(optimizer)

        model.compile(loss=["binary_crossentropy"], optimizer=optimizer, metrics=[])
        print("model compiled", hvd.rank())

    callbacks = [hvd.callbacks.BroadcastGlobalVariablesCallback(0)]

    if hvd.rank() == 0:
        if args.checkpoint_interval > 0:
            callbacks.append(ModelCheckpoint(args.output_file + ".checkpoint-{epoch}",  period=args.checkpoint_interval))

        if args.print_lr:
            callbacks.append(PrintLR(model))

        callbacks.append(Metrics(model))

        print(model.summary(line_length=118))
        print("Number of GPUs in use:", hvd.size())
        print("Batch size:", args.batch_size)
        print("Learning rate:", K.eval(model.optimizer.lr))
        # print("Dropout:", args.dropout)

    if args.patience > -1:
        callbacks.append(EarlyStopping(patience=args.patience, verbose=1))
    
    print("Beginning fit", hvd.rank())

    model.fit_generator(data_generator(args.train, args.batch_size, seq_len=args.seq_len),
                        steps_per_epoch=ceil( get_example_count(args.train) / args.batch_size / hvd.size() ),
                        use_multiprocessing=True, epochs=args.epochs, callbacks=callbacks,
                        validation_data=data_generator(args.dev, args.eval_batch_size, seq_len=args.seq_len),
                        validation_steps=ceil( get_example_count(args.dev) / args.eval_batch_size / hvd.size() ),
                        verbose=1 if hvd.rank() == 0 else 0)

    if hvd.rank() == 0:
        print("Saving model:", args.output_file)
        model.save(args.output_file)

if __name__ == "__main__":

    args = argparser()
    build_model(args)
