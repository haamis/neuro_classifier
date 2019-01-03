import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
from keras.preprocessing import sequence
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Activation
from keras.layers import Embedding, Input
from keras.layers import Conv1D, GlobalMaxPooling1D

import keras_metrics

from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer

import sys
import json
import re
import pickle
from tqdm import tqdm

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
set_session(tf.Session(config=config))

# set parameters:
max_features = 5000
maxlen = 5000
batch_size = 128
embedding_dims = 300
filters = 250
kernel_size = 3
epochs = 2

def transform(input_file):
    print("Reading input file..")

    with open("./" + input_file) as f:
        data = json.load(f)

    abstracts = []
    is_neuro = []

    # Neuroscience MeSH-terms.
    reg = re.compile(r"(D009420|D009457|D009474|D009422|D001520|D011579|D001523|D004191)")

    for article in tqdm(data, desc="Grabbing abstracts and mesh terms"):
        abstract_parts = []
        abstracts.append("\n".join([x["text"] for x in article["abstract"]]))
        is_neuro.append("no")
        for mesh_term in article["mesh_list"]:
            if reg.match(mesh_term["mesh_id"]):
                is_neuro[-1] = "yes"
                break

    del data # Release the huge chunk of memory, ugly. TODO: structure code properly.

    print("\n",abstracts[0])
    print(is_neuro[0])

    print("Running vectorizer.")
    vectorizer = CountVectorizer(max_features=max_features, binary=True, ngram_range=(1,1))
    abstracts = vectorizer.fit_transform(abstracts)
    print("abstract shape:", abstracts.shape)
    print("abstracts[0]:", abstracts[0])

    label_encoder = LabelEncoder()
    is_neuro = label_encoder.fit_transform(is_neuro)
    print("is_neuro shape", is_neuro.shape)
    print("is_neuro label_encoder", is_neuro)
    print("is_neuro class labels", label_encoder.classes_)

    one_hot_encoder = OneHotEncoder(sparse=False)
    is_neuro = one_hot_encoder.fit_transform(is_neuro.reshape(-1,1))
    print("is_neuro 1hot shape", is_neuro.shape)
    print("is_neuro 1hot", is_neuro)

    return (abstracts, is_neuro)


def build_model():

    #abstracts = sequence.pad_sequences(abstracts.toarray(), maxlen=abstracts.shape[0])

    # Let's define the inputs
    x = Input(shape=(maxlen,))

    # we start off with an efficient embedding layer which maps
    # our vocab indices into embedding_dims dimensions

    embedding_layer = Embedding(max_features,
                        embedding_dims,
                        input_length=maxlen)

    embeddings = embedding_layer(x)

    conv_layer = Conv1D(filters, kernel_size, padding='valid', activation='relu', strides=1)
    conv_result = conv_layer(embeddings)
    pooled = (GlobalMaxPooling1D())(conv_result) 

    # We add a vanilla hidden layer:
    out = Dense(2, activation='softmax')(pooled)

    model = Model(x, out)

    #sample_weights = tf.multiply(is_neuro, tf.constant([0.02, 1.0], dtype='float64'))
    #weights = tf.gather(tf.constant([0.02, 1.0], dtype='float64'), tf.constant(is_neuro, dtype='int32'))


    #loss=tf.nn.weighted_cross_entropy_with_logits(is_neuro, out, weights)
    model.compile(loss='categorical_crossentropy',
                optimizer='adam',
                metrics=[keras_metrics.precision(), keras_metrics.recall()])

    print(model.summary())

    for epoch in range(epochs):

        model_hist = model.fit(abstracts, is_neuro,
                batch_size=batch_size,
                epochs=1,
                validation_split=0.2,
                class_weight={0: 0.02, 1:1.0})
        precision = model_hist.history['val_precision'][0]
        recall = model_hist.history['val_recall'][0]
        f_score = (2.0 * precision * recall) / (precision + recall)
        print(f_score)

    return model


def dump_data(file_name, dump_data):

    with open(file_name, "wb") as f:
        pickle.dump(dump_data, f)


def load_data(file_name):
    
    with open(file_name, "rb") as f:
        return pickle.load(f)


(abstracts, is_neuro) = transform(sys.argv[1])
#dump_data("../abstracts_dump", abstracts)
#dump_data("../is_neuro_dump", is_neuro)

#abstracts = load_data("../abstracts_dump")
#is_neuro = load_data("../is_neuro_dump")

model = build_model()
#dump_data("../model_dump", model)

model = load_data("../model_dump")

