import csv, os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

try:
    import ujson as json
except ImportError:
    import json

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
from keras.layers import Average, Maximum, Concatenate, Conv1D, Dense, Dropout, Flatten, GlobalAveragePooling1D, GlobalMaxPooling1D, Lambda, Reshape, Permute
from keras.models import Model
from keras.optimizers import Adam
from keras.utils import multi_gpu_model

from keras_bert.loader import load_trained_model_from_checkpoint
from keras_bert import AdamWarmup, calc_train_steps
from keras_bert.activations import gelu
from keras_multi_head import MultiHeadAttention, MultiHead
from keras_self_attention import SeqSelfAttention
from keras_position_wise_feed_forward import FeedForward
from keras_transformer import get_encoder_component


def argparser():
    arg_parse = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    arg_parse.add_argument("--train", help="Processed train file.", metavar="FILE", required=True)
    arg_parse.add_argument("--dev", help="Processed dev file.", metavar="FILE", required=True)
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
    arg_parse.add_argument("--bp_index", help="List of BP indices in labels.", metavar="FILE", required=True)
    arg_parse.add_argument("--cc_index", help="List of CC indices in labels.", metavar="FILE", required=True)
    arg_parse.add_argument("--mf_index", help="List of MF indices in labels.", metavar="FILE", required=True)
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

def get_sub_ontology_size(file_path):
    return len(json.load(xopen(file_path)))

def data_generator(file_path, batch_size, seq_len=512):
    bp_index = json.load(xopen(args.bp_index))
    cc_index = json.load(xopen(args.cc_index))
    mf_index = json.load(xopen(args.mf_index))
    while True:
        with xopen(file_path, "rt") as f:
            cr = csv.reader(f, delimiter="\t")
            next(cr) # Skip example number row.
            text = []
            labels = []
            for line in cr:
                if len(text) == batch_size:
                    # Fun fact: the 2 inputs must be in a list, *not* a tuple. Why.
                    yield ([np.asarray(text), np.zeros_like(text)], [np.asarray([l[bp_index] for l in labels]),
                    np.asarray([l[cc_index] for l in labels]), np.asarray([l[mf_index] for l in labels])])
                    text = []
                    labels = []
                text.append(np.asarray(json.loads(line[0]))[0:seq_len])
                labels.append(np.asarray(json.loads(line[1])))
            # Yield what is left as the last batch when file has been read to its end.
            yield ([np.asarray(text), np.zeros_like(text)], [np.asarray([l[bp_index] for l in labels]),
                    np.asarray([l[cc_index] for l in labels]), np.asarray([l[mf_index] for l in labels])])

class Metrics(Callback):

    def __init__(self):

        self.best_f1 = {"BP": 0, "CC":0, "MF":0}
        self.best_f1_epoch = {"BP": 0, "CC":0, "MF":0}
        self.best_f1_threshold = {"BP": 0, "CC":0, "MF":0}
        self.all_labels = []
        if args.label_mapping:
            print("in if")
            file_name = args.dev_all
        else:
            print("in else")
            file_name = args.dev
        with xopen(file_name, "rt") as f:
            cr = csv.reader(f, delimiter="\t")
            next(cr) # Skip example number row.
            for line in tqdm(cr, desc="Reading dev labels"):
                self.all_labels.append(json.loads(line[1]))
            self.all_labels = lil_matrix(self.all_labels, dtype='b')
            # print("Dev labels shape:", self.all_labels.shape)
            self.bp_labels = self.all_labels[:, json.load(xopen(args.bp_index))]
            self.cc_labels = self.all_labels[:, json.load(xopen(args.cc_index))]
            self.mf_labels = self.all_labels[:, json.load(xopen(args.mf_index))]
            print("BP labels shape:", self.bp_labels.shape)
            print("CC labels shape:", self.cc_labels.shape)
            print("MF labels shape:", self.mf_labels.shape)

        if args.dev_all is not None:
            with xopen(args.label_mapping) as f:
                self.labels_mapping = json.loads(f.read())

    def on_epoch_end(self, epoch, logs=None):
        print("Predicting probabilities..")
        labels_prob = self.model.predict_generator(data_generator(args.dev, args.eval_batch_size, seq_len=args.seq_len), use_multiprocessing=True,
                                                    steps=ceil(get_example_count(args.dev) / args.eval_batch_size), verbose=1)
        print("Probabilities to labels..\n")
        if args.label_mapping is not None:
            full_labels_prob = np.zeros(self.all_labels.shape)
            for i, probs in enumerate(labels_prob):
               np.put(full_labels_prob[i], self.labels_mapping, probs)

            labels_prob = full_labels_prob

        for sub_labels_prob, ontology, labels in zip(labels_prob, ["BP", "CC", "MF"], [self.bp_labels, self.cc_labels, self.mf_labels]):
            print(ontology,'\n')
            for threshold in np.arange(args.threshold_start, args.threshold_end, args.threshold_step):
                print("Threshold:", threshold)
                labels_pred = lil_matrix(sub_labels_prob.shape, dtype='b')
                labels_pred[sub_labels_prob>threshold] = 1
                precision, recall, f1, _ = precision_recall_fscore_support(labels, labels_pred, average="micro")
                if f1 > self.best_f1[ontology]:
                    self.best_f1[ontology] = f1
                    self.best_f1_epoch[ontology] = epoch
                    self.best_f1_threshold[ontology] = threshold
                print("Precision:", precision)
                print("Recall:", recall)
                print("F1-score:", f1, "\n")

            print("Current", ontology, "F_max:", self.best_f1[ontology], "epoch", self.best_f1_epoch[ontology]+1, "threshold", self.best_f1_threshold[ontology], '\n')


def loss_fn(y_true, y_pred):
    return keras.losses.binary_crossentropy(y_true, y_pred) + keras.losses.categorical_hinge(y_true, y_pred)

def build_model(args):

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    #config.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1
    K.set_session(tf.Session(config=config))

    print("Building model..")
    bert = load_trained_model_from_checkpoint(args.bert_config, args.init_checkpoint,
                                                training=False, trainable=True,
                                                seq_len=args.seq_len)

    #slice_layer = Lambda(lambda x: K.slice(x, [0, 0, 0], [-1, 1, -1]))(bert.get_layer("Encoder-12-FeedForward-Norm").output)

    # slices = []
    # for i in range(1, 13):
    #     layer_name = "Encoder-" + str(i) + "-FeedForward-Norm"
    #     cls_slice = Lambda(lambda x: K.slice(x, [0, 0, 0], [-1, 1, -1]))(bert.get_layer(layer_name).output)
    #     # Simple layer to drop masking as it does not pass it along with layer.compute_mask().
    #     slices.append(Lambda(lambda x: x)(cls_slice))

    # cls_average = Average()(slices)
    # # This reshape might seem odd, but is required.
    # slices_shaped = Reshape((768,))(cls_average)
    # print(slices_shaped.shape)

    # Simple layer to drop masking as it does not pass it along with layer.compute_mask().
    # drop_mask = Lambda(lambda x: x)(bert.get_layer("Encoder-12-FeedForward-Norm").output)

    # conv1 = Conv1D(250, 25, padding='valid', activation='relu', strides=1)(drop_mask)
    # conv2 = Conv1D(250, 10, padding='valid', activation='relu', strides=1)(drop_mask)
    # conv3 = Conv1D(250, 3, padding='valid', activation='relu', strides=1)(drop_mask)
    # conv4 = Conv1D(250, 1, padding='valid', activation='relu', strides=1)(drop_mask)

    # max_pool1 = GlobalMaxPooling1D()(conv1)
    # max_pool2 = GlobalMaxPooling1D()(conv2)
    # max_pool3 = GlobalMaxPooling1D()(conv3)
    # max_pool4 = GlobalMaxPooling1D()(conv4)

    # concat = Concatenate()([max_pool1, max_pool2, max_pool3, max_pool4])

    #attention_layer = MultiHeadAttention(head_num=12)(drop_mask)

    transformer_output_bp = get_encoder_component(name="Encoder-BP", input_layer=bert.layers[-1].output,
                                            head_num=12, hidden_dim=3072, feed_forward_activation=gelu,
                                            dropout_rate=0.1)
    drop_mask_bp = Lambda(lambda x: x, name="drop_mask_BP")(transformer_output_bp)
    slice_CLS_bp = Lambda(lambda x: K.slice(x, [0, 0, 0], [-1, 1, -1]), name="slice_CLS_BP")(drop_mask_bp)
    flatten_CLS_bp = Flatten()(slice_CLS_bp)
    # Needed to avoid a json serialization error when saving the model.
    last_position = args.seq_len-1
    slice_SEP_bp = Lambda(lambda x: K.slice(x, [0, last_position, 0], [-1, 1, -1]), name="slice_SEP_BP")(drop_mask_bp)
    flatten_SEP_bp = Flatten()(slice_SEP_bp)
    permute_layer_bp = Permute((2, 1))(drop_mask_bp)
    permute_average_bp = GlobalAveragePooling1D()(permute_layer_bp)
    permute_maximum_bp =  GlobalMaxPooling1D()(permute_layer_bp)
    concat_bp = Concatenate()([permute_average_bp, permute_maximum_bp, flatten_CLS_bp, flatten_SEP_bp])
    output_layer_bp = Dense(get_sub_ontology_size(args.bp_index), activation='sigmoid', name="label_BP")(concat_bp)

    transformer_output_cc = get_encoder_component(name="Encoder-CC", input_layer=bert.layers[-1].output,
                                            head_num=12, hidden_dim=3072, feed_forward_activation=gelu,
                                            dropout_rate=0.1)
    drop_mask_cc = Lambda(lambda x: x, name="drop_mask_CC")(transformer_output_cc)
    slice_CLS_cc = Lambda(lambda x: K.slice(x, [0, 0, 0], [-1, 1, -1]), name="slice_CLS_CC")(drop_mask_cc)
    flatten_CLS_cc = Flatten()(slice_CLS_cc)
    # Needed to avoid a json serialization error when saving the model.
    last_position = args.seq_len-1
    slice_SEP_cc = Lambda(lambda x: K.slice(x, [0, last_position, 0], [-1, 1, -1]), name="slice_SEP_CC")(drop_mask_cc)
    flatten_SEP_cc = Flatten()(slice_SEP_cc)
    permute_layer_cc = Permute((2, 1))(drop_mask_cc)
    permute_average_cc = GlobalAveragePooling1D()(permute_layer_cc)
    permute_maximum_cc =  GlobalMaxPooling1D()(permute_layer_cc)
    concat_cc = Concatenate()([permute_average_cc, permute_maximum_cc, flatten_CLS_cc, flatten_SEP_cc])
    output_layer_cc = Dense(get_sub_ontology_size(args.cc_index), activation='sigmoid', name="label_cc")(concat_cc)

    transformer_output_mf = get_encoder_component(name="Encoder-MF", input_layer=bert.layers[-1].output,
                                            head_num=12, hidden_dim=3072, feed_forward_activation=gelu,
                                            dropout_rate=0.1)
    drop_mask_mf = Lambda(lambda x: x, name="drop_mask_MF")(transformer_output_mf)
    slice_CLS_mf = Lambda(lambda x: K.slice(x, [0, 0, 0], [-1, 1, -1]), name="slice_CLS_MF")(drop_mask_mf)
    flatten_CLS_mf = Flatten()(slice_CLS_mf)
    # Needed to avoid a json serialization error when saving the model.
    last_position = args.seq_len-1
    slice_SEP_mf = Lambda(lambda x: K.slice(x, [0, last_position, 0], [-1, 1, -1]), name="slice_SEP_MF")(drop_mask_mf)
    flatten_SEP_mf = Flatten()(slice_SEP_mf)
    permute_layer_mf = Permute((2, 1))(drop_mask_mf)
    permute_average_mf = GlobalAveragePooling1D()(permute_layer_mf)
    permute_maximum_mf =  GlobalMaxPooling1D()(permute_layer_mf)
    concat_mf = Concatenate()([permute_average_mf, permute_maximum_mf, flatten_CLS_mf, flatten_SEP_mf])
    output_layer_mf = Dense(get_sub_ontology_size(args.mf_index), activation='sigmoid', name="label_mf")(concat_mf)

    model = Model(bert.input, [output_layer_bp, output_layer_cc, output_layer_mf])

    if args.gpus > 1:
        template_model = model
        model = multi_gpu_model(template_model, gpus=args.gpus)

    callbacks = [Metrics()]
    # CyclicLR(5e-5, 1e-4, 10000, "triangular2")

    if args.patience > -1:
        callbacks.append(EarlyStopping(patience=args.patience, verbose=1))

    if args.checkpoint_interval > 0:
        callbacks.append(ModelCheckpoint(args.output_file + ".checkpoint-{epoch}",  period=args.checkpoint_interval))

    total_steps, warmup_steps =  calc_train_steps(num_example=get_example_count(args.train),
                                                batch_size=args.batch_size, epochs=args.epochs,
                                                warmup_proportion=0.01)

    optimizer = AdamWarmup(total_steps, warmup_steps, lr=args.lr)
    # optimizer = Adam(args.lr)

    model.compile(loss="binary_crossentropy", optimizer=optimizer, metrics=["accuracy"])

    print(model.summary(line_length=118))
    print("Number of GPUs in use:", args.gpus)
    print("Batch size:", args.batch_size)
    print("Learning rate:", args.lr)
    print("Dropout:", args.dropout)


    model.fit_generator(data_generator(args.train, args.batch_size, seq_len=args.seq_len),
                        steps_per_epoch=ceil( get_example_count(args.train) / args.batch_size ),
                        use_multiprocessing=True, epochs=args.epochs, callbacks=callbacks,
                        validation_data=data_generator(args.dev, args.eval_batch_size, seq_len=args.seq_len),
                        validation_steps=ceil( get_example_count(args.dev) / args.eval_batch_size ))

    print("Saving model:", args.output_file)
    if args.gpus > 1:
        template_model.save(args.output_file)
    else:
        model.save(args.output_file)

if __name__ == "__main__":

    args = argparser()
    build_model(args)
