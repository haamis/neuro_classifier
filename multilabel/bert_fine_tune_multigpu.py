import csv, json, os
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

from keras_contrib.callbacks import CyclicLR


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

class Metrics(Callback):

    def __init__(self):

        self.best_f1 = 0
        self.best_f1_epoch = 0
        self.best_f1_threshold = 0

        #print("Metrics init, reading dev labels..")
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
        print("Predicting probabilities..")
        labels_prob = self.model.predict_generator(data_generator(args.dev, args.eval_batch_size, seq_len=args.seq_len), use_multiprocessing=True,
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
            labels_pred[labels_prob>threshold] = 1
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

    transformer_output = get_encoder_component(name="Encoder-13", input_layer=bert.layers[-1].output,
                                            head_num=12, hidden_dim=3072, feed_forward_activation=gelu,
                                            dropout_rate=0.1)

    # transformer_output2 = get_encoder_component(name="Encoder-14", input_layer=transformer_output,
    #                                        head_num=12, hidden_dim=3072, feed_forward_activation=gelu,
    #                                        dropout_rate=0.1)

    # transformer_output3 = get_encoder_component(name="Encoder-15", input_layer=transformer_output2,
    #                                        head_num=12, hidden_dim=3072, feed_forward_activation=gelu)

    drop_mask = Lambda(lambda x: x, name="drop_mask")(transformer_output)#(bert.layers[-1].output)

    slice_CLS = Lambda(lambda x: K.slice(x, [0, 0, 0], [-1, 1, -1]), name="slice_CLS")(drop_mask)
    flatten_CLS = Flatten()(slice_CLS)

    # Needed to avoid a json serialization error when saving the model.
    last_position = args.seq_len-1
    slice_SEP = Lambda(lambda x: K.slice(x, [0, last_position, 0], [-1, 1, -1]), name="slice_SEP")(drop_mask)
    flatten_SEP = Flatten()(slice_SEP)

    permute_layer = Permute((2, 1))(drop_mask)

    all_average = GlobalAveragePooling1D()(drop_mask)
    #cls_average2 = GlobalAveragePooling1D()(drop_mask)
    permute_average = GlobalAveragePooling1D()(permute_layer)

    all_maximum =  GlobalMaxPooling1D()(drop_mask)
    #cls_maximum2 =  GlobalMaxPooling1D()(drop_mask)
    permute_maximum =  GlobalMaxPooling1D()(permute_layer)

    concat = Concatenate()([permute_average, permute_maximum, flatten_CLS, flatten_SEP])

    #slice_layer3 = Lambda(lambda x: K.slice(x, [0, 0, 0], [-1, 128, -1]), name="slice_128")(drop_mask)

    #flatten_layer = Flatten()(slice_layer3)

    #avg_pool = GlobalAveragePooling1D()(drop_mask)

    #dropout_layer = Dropout(args.dropout)(drop_mask)

    # TODO: try concat of avg and max pools of both CLS and the permutation

    output_layer = Dense(get_label_dim(args.train), activation='sigmoid', name="label_out")(concat)

    model = Model(bert.input, output_layer)

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

    model.compile(loss=["binary_crossentropy"], optimizer=optimizer, metrics=["accuracy"])

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
