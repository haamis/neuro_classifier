import pickle, sys
import tensorflow as tf
import numpy as np
import keras.backend as K

#from timeit import default_timer as timer

from scipy.sparse import lil_matrix

from keras.backend.tensorflow_backend import set_session
from keras.layers import Bidirectional, Concatenate, Conv1D, Dense, Dropout, Input, GlobalMaxPooling1D, Lambda
from keras.models import Model
from keras.preprocessing import sequence, text
from keras.optimizers import Adam, SGD
from keras.layers import CuDNNGRU as GRU
from keras.layers import CuDNNLSTM as LSTM
from keras.callbacks import Callback

from keras_bert.loader import load_trained_model_from_checkpoint
from keras_bert.bert import *

from bert import tokenization

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import precision_score, recall_score, f1_score, precision_recall_fscore_support

from tqdm import tqdm

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
set_session(tf.Session(config=config))

# set parameters:
batch_size = 64
max_batch_size = 64
filters = 250
kernel_size = 3
epochs = 1000
maxlen = 384


def dump_data(file_name, data):

    with open(file_name, "wb") as f:
        pickle.dump(data, f)

def load_data(file_name):

    with open(file_name, "rb") as f:
        return pickle.load(f)

def tokenize(abstracts, maxlen=512):
    tokenizer = tokenization.FullTokenizer("../../biobert_pubmed/vocab.txt")
    ret_val = []
    for abstract in abstracts:
        abstract = ["[CLS]"] + tokenizer.tokenize(abstract[0:maxlen-2]) + ["[SEP]"]
        ret_val.append(abstract)
    return ret_val, tokenizer.vocab, tokenizer.inv_vocab

def transform(abstracts_file, mesh_file):

    print("Reading input files..")

    abstracts = load_data("./" + abstracts_file)
    labels = load_data("./" + mesh_file)

    print("Tokenizing..")
    abstracts, vocab, inv_vocab = tokenize(abstracts, maxlen=maxlen)

    print("Vectorizing..")
    token_vectors = np.asarray( [np.asarray( [vocab[token] for token in abstract] + [0] * (maxlen - len(abstract)) ) for abstract in abstracts] )
    del abstracts
    print("Token_vectors shape:", token_vectors.shape)

    print("Binarizing labels..")
    mlb = MultiLabelBinarizer(sparse_output=True)
    labels = mlb.fit_transform(labels)
    labels = labels.astype('b')
    print("Labels shape:", labels.shape)
    print(np.dtype(labels))

    print("Splitting..")
    token_vectors_train, token_vectors_test, labels_train, labels_test = train_test_split(token_vectors, labels, test_size=0.1)

    _, sequence_len = token_vectors_train.shape

    return token_vectors_train, token_vectors_test, labels_train, labels_test, sequence_len, inv_vocab


def build_model(abstracts_train, abstracts_test, labels_train, labels_test, sequence_len, inv_vocab):

    #input_layer = Input(shape=(sequence_len,))

    checkpoint_file = "../../biobert_pubmed/biobert_model.ckpt"
    config_file = "../../biobert_pubmed/bert_config.json"

    #dummy_input = Input(tensor=tf.zeros(512))

    biobert = load_trained_model_from_checkpoint(config_file, checkpoint_file, training=False, seq_len=sequence_len)
    biobert_train = load_trained_model_from_checkpoint(config_file, checkpoint_file, training=True, seq_len=sequence_len)

    #extract = biobert_train.layers[-6](biobert.layers[-1].output)

    #print(biobert.summary())
    #print(biobert_train.summary())

    print(biobert.input)
    print(biobert.layers[-1].output)

    # Drop mask from bert output.
    #drop_mask_layer = Lambda(lambda x, mask: x)(biobert.layers[-1].output)

    #rnn_layer1 = Bidirectional(LSTM(768, return_sequences=True))(drop_mask_layer)

    #rnn_layer2 = Bidirectional(LSTM(768, return_sequences=True))(rnn_layer1)

    #rnn_layer3 = Bidirectional(LSTM(768, return_sequences=False))(rnn_layer2)

    # conv_layer1 = Conv1D(250, 3, activation='relu')(drop_mask_layer)
    # pooled = GlobalMaxPooling1D()(conv_layer1)

    # conv_res = []
    # for width in range(2,5):

    #     conv_result = Conv1D(filters, width, padding='valid', activation='relu', strides=1)(drop_mask_layer)
    #     pooled = GlobalMaxPooling1D()(conv_result)
    #     conv_res.append(pooled)

    # concatenated = Concatenate()(conv_res)

    #meme = Dense(768, activation='tanh')(extract)

    print(tf.gather_nd(biobert.layers[-1].output, [[[0]]]))

    output_layer = Dense(labels_train.shape[1], activation='sigmoid')(tf.gather_nd(biobert.layers[-1].output, [[[0]]]))

    model = Model(biobert.input, output_layer)

    print(model.summary(line_length=118))

    learning_rate = 0.001

    model.compile(loss='binary_crossentropy',
                optimizer=SGD(lr=0.01, momentum=0.9, nesterov=True))

    #print(model.summary(line_length=115))

    for epoch in range(epochs):
        print("Epoch", epoch + 1)
        #learning_rate -= 0.0001
        cur_batch_size = min(batch_size + int(0.125 * epoch * batch_size), max_batch_size)
        print("batch size:", cur_batch_size)
        # model.compile(loss='binary_crossentropy',
        #         optimizer=Adam(lr=learning_rate))
        print("learning rate:", K.eval(model.optimizer.lr))
        model.fit([abstracts_train, np.zeros_like(abstracts_train)], labels_train,
            batch_size=cur_batch_size,
            epochs=1,
            validation_data=[[abstracts_test, np.zeros_like(abstracts_test)], labels_test])
        print("Predicting probabilities..")
        labels_prob = model.predict([abstracts_test, np.zeros_like(abstracts_test)])

        print("Probabilities to labels..")
        labels_pred = lil_matrix(labels_prob.shape, dtype='b')
        labels_pred[labels_prob>0.5] = 1

        precision, recall, f1, _ = precision_recall_fscore_support(labels_test, labels_pred, average='micro')
        print("Precision:", precision)
        print("Recall:", recall)
        print("F1-score:", f1, "\n")

build_model(*transform(sys.argv[1], sys.argv[2]))
