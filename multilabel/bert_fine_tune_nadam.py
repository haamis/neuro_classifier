import pickle, sys
import tensorflow as tf
import numpy as np
import keras.backend as K

from scipy.sparse import lil_matrix

from keras.backend.tensorflow_backend import set_session
from keras.models import load_model
from keras.optimizers import Adam, Adamax, SGD
from keras.utils import multi_gpu_model

#from keras_bert.bert import *
from keras_bert import get_custom_objects

from bert import tokenization

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import precision_recall_fscore_support

from tqdm import tqdm

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
set_session(tf.Session(config=config))

# set parameters:
gpus = 1
batch_size = 5
epochs = 20
maxlen = 512
freeze_bert = False


def dump_data(file_name, data):

    with open(file_name, "wb") as f:
        pickle.dump(data, f)

def load_data(file_name):

    with open(file_name, "rb") as f:
        return pickle.load(f)

def tokenize(abstracts, maxlen=512):
    tokenizer = tokenization.FullTokenizer("../../biobert_pubmed/vocab.txt", do_lower_case=False)
    ret_val = []
    for abstract in tqdm(abstracts,desc="Tokenizing abstracts"):
        abstract = ["[CLS]"] + tokenizer.tokenize(abstract[0:maxlen-2]) + ["[SEP]"]
        ret_val.append(abstract)
    return ret_val, tokenizer.vocab

def transform(abstracts_file, mesh_file):

    print("Reading input files..")

    abstracts = load_data("./" + abstracts_file)
    labels = load_data("./" + mesh_file)

    abstracts, vocab = tokenize(abstracts, maxlen=maxlen)

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

    return token_vectors_train, token_vectors_test, labels_train, labels_test


def build_model(abstracts_train, abstracts_test, labels_train, labels_test):

    custom_objects = get_custom_objects()
    custom_objects["tf"] = tf

    if gpus > 1:
        base_model = load_model(sys.argv[3], custom_objects=custom_objects)
        model = multi_gpu_model(base_model, gpus=gpus, cpu_merge=True, cpu_relocation=False)
    else:
        model = load_model(sys.argv[3], custom_objects=custom_objects)

    # Unfreeze bert layers.
    if not freeze_bert:
        for layer in model.layers[:]:
            layer.trainable = True

    print(model.summary(line_length=118))

    print("Number of GPUs in use:", gpus)
    
    learning_rate = 0.00005
    
    model.compile(loss='binary_crossentropy',
                optimizer=Adamax(lr=learning_rate))#SGD(lr=0.2, momentum=0.9))
    
    best_f1 = 0.0
    stale_epochs = 0

    for epoch in range(epochs):
        print("Epoch", epoch + 1)
        print("batch size:", batch_size)
        print("learning rate:", K.eval(model.optimizer.lr))
        model.fit([abstracts_train, np.zeros_like(abstracts_train)], labels_train,
            batch_size=batch_size,
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
        
        if f1 > best_f1:
            best_f1 = f1
            stale_epochs = 0
            print("Saving model..\n")
            if gpus > 1:
                base_model.save(sys.argv[4])
            else:
                model.save(sys.argv[4])
        else:
            stale_epochs += 1
            if stale_epochs >= 4:
                break

build_model(*transform(sys.argv[1], sys.argv[2]))
