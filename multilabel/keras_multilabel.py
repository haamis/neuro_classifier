import pickle, sys
import tensorflow as tf
import numpy as np

from keras.backend.tensorflow_backend import set_session
from keras.layers import (Dense, Input)
from keras.models import Model

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import f1_score

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
set_session(tf.Session(config=config))

# set parameters:
batch_size = 1024
filters = 250
epochs = 25

def transform(abstracts_file, mesh_file):
    print("Reading input files..")

    abstracts = load_data("./" + abstracts_file)
    labels = load_data("./" + mesh_file)

    print("Running vectorizer..")
    vectorizer = TfidfVectorizer(ngram_range=(1,1))
    abstracts = vectorizer.fit_transform(abstracts)
    print("abstract shape:", abstracts.shape)
    print("abstracts[0]:", abstracts[0][0])

    print("Binarizing labels..")
    mlb = MultiLabelBinarizer()
    labels = mlb.fit_transform(labels)
    
    print("Splitting..")
    abstracts_train, abstracts_test, labels_train, labels_test = train_test_split(abstracts, labels, test_size=0.2)

    return (abstracts_train, labels_train, abstracts_test, labels_test)


def build_model():

    feature_count = abstracts_train.shape[1]

    input_layer = Input(shape=(feature_count,))

    hidden = Dense(500, activation="relu")(input_layer)

    out = Dense(labels_train.shape[1], activation='sigmoid')(hidden)

    model = Model(input_layer, out)

    model.compile(loss='binary_crossentropy',
                optimizer='adam')

    print(model.summary())

    for epoch in range(epochs):

        model.fit(abstracts_train, labels_train,
                batch_size=batch_size,
                epochs=1)
        pred_labels = model.predict(abstracts_test)
        print(f1_score(labels_test, pred_labels, average='micro'))
        #precision = model_hist.history['val_precision'][0]
        #recall = model_hist.history['val_recall'][0]
        #f_score = (2.0 * precision * recall) / (precision + recall)
        #print("epoch", epoch + 1, "F-score:", f_score, "\n")

    return model
    

def dump_data(file_name, data):

    with open(file_name, "wb") as f:
        pickle.dump(data, f)


def load_data(file_name):
    
    with open(file_name, "rb") as f:
        return pickle.load(f)


(abstracts_train, labels_train, abstracts_test, labels_test) = transform(sys.argv[1], sys.argv[2])

#abstracts = load_data("../abstracts_dump")
#is_neuro = load_data("../is_neuro_dump")

model = build_model()
#dump_data("../model_dump", model)

#model = load_data("../model_dump")
