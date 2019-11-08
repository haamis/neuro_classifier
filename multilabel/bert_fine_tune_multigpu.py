import pickle, sys, lzma
import tensorflow as tf
import keras.backend as K

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

from scipy.sparse import lil_matrix

from keras.callbacks import Callback, EarlyStopping, ModelCheckpoint
from keras.layers import Dense, Flatten, Lambda, Dropout
from keras.models import Model
from keras.utils import multi_gpu_model

from keras_bert.loader import load_trained_model_from_checkpoint
from keras_bert import AdamWarmup, calc_train_steps

from sklearn.metrics import precision_recall_fscore_support

# set parameters:
batch_size = 5
gpus = 1
learning_rate = 4e-5
epochs = 15
maxlen = 512

def argparser():
    arg_parse = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    arg_parse.add_argument("--input_file", help="TSV input file.", metavar="FILE", required=True)
    arg_parse.add_argument("--init_checkpoint", help="BERT tensorflow model with path ending in .ckpt", metavar="PATH", required=True)
    arg_parse.add_argument("--output_model", help="Path to which save the finetuned model.", metavar="PATH", required=True)
    arg_parse.add_argument("--vocab", help="Vocabulary to use.", metavar="FILE", required=True)
    arg_parse.add_argument("--batch_size", help="Batch size to use for finetuning.", metavar="INT", type=int, required=True)
    arg_parse.add_argument("--lr", "--learning_rate", help="Peak learning rate.", metavar="FLOAT", type=float, required=True)
    arg_parse.add_argument("--epochs", help="Max amount of epochs to run.", metavar="INT", type=int, required=True)
    arg_parse.add_argument("--seq_len", help="BERT's maximum sequence length", metavar="INT", default=512, type=int)
    arg_parse.add_argument("--gpus", help="Number of GPUs to use,", metavar="INT", default=1, type=int)
    arg_parse.add_argument("--patience", help="Patience of early stopping. Early stopping disabled if 0.", metavar="INT", default=0, type=int)
    arg_parse.add_argument("--threshold", help="Positive label prediction threshold.", metavar="FLOAT", default=0.5, type=float)
    return arg_parse

class Metrics(Callback):

    def on_epoch_end(self, batch, logs={}):
        print("Predicting probabilities..")
        labels_prob = self.model.predict([abstracts_test, lil_matrix(abstracts_test.shape)])

        print("Probabilities to labels..")
        labels_pred = lil_matrix(labels_prob.shape, dtype='b')
        labels_pred[labels_prob>0.5] = 1

        precision, recall, f1, _ = precision_recall_fscore_support(labels_test, labels_pred, average="micro")
        print("Precision:", precision)
        print("Recall:", recall)
        print("F1-score:", f1, "\n")

def load_data(file_name):

    with lzma.open(file_name, "rb") as f:
        return pickle.load(f)

def build_model(abstracts_train, abstracts_test, labels_train, labels_test, sequence_len):

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    #config.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1
    K.set_session(tf.Session(config=config))

    checkpoint_file = "../../biobert_pubmed/biobert_model.ckpt"
    config_file = "../../biobert_pubmed/bert_config.json"

    biobert = load_trained_model_from_checkpoint(config_file, checkpoint_file,
                                                training=False, trainable=True,
                                                seq_len=sequence_len)

    slice_layer = Lambda(lambda x: K.slice(x, [0, 0, 0], [-1, 1, -1]))(biobert.layers[-1].output)

    flatten_layer = Flatten()(slice_layer)

    dropout_layer = Dropout(0.1)(flatten_layer)

    output_layer = Dense(labels_train.shape[1], activation='sigmoid')(dropout_layer)

    model = Model(biobert.input, output_layer)

    if gpus > 1:
        model = multi_gpu_model(model, gpus=gpus)

    metrics = Metrics()
    # --patience arg parser
    early_stopping = EarlyStopping(patience=15, verbose=1)

    total_steps, warmup_steps = calc_train_steps(num_example=abstracts_train.shape[0],
                                                batch_size=batch_size, epochs=epochs,
                                                warmup_proportion=0.1)

    optimizer = AdamWarmup(total_steps, warmup_steps, lr=learning_rate)

    model.compile(loss="binary_crossentropy", optimizer=optimizer)
    
    print(model.summary(line_length=118))
    print("Number of GPUs in use:", gpus)
    print("Batch size:", batch_size)
    print("Learning rate:", learning_rate)

    model.fit([abstracts_train, lil_matrix(abstracts_train.shape)], labels_train,
        batch_size=batch_size, epochs=epochs, callbacks=[metrics, ModelCheckpoint(sys.argv[5][:-3] + "-checkpoint-{epoch}.h5")],
        validation_data=[[abstracts_test, lil_matrix(abstracts_test.shape)], labels_test])
    model.save(sys.argv[5])

if __name__ == "__main__":

    args = argparser().parse_args()

    print("Reading input files..")

    # abstracts_train = load_data(sys.argv[1])[0:1000]
    # abstracts_test = load_data(sys.argv[2])[0:100]
    # labels_train = load_data(sys.argv[3])[0:1000]
    # labels_test = load_data(sys.argv[4])[0:100]

    _, sequence_len = abstracts_train.shape

    build_model(abstracts_train, abstracts_test, labels_train, labels_test, sequence_len)
