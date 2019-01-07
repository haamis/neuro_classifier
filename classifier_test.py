import json, pickle, re, sys
import keras_metrics
import tensorflow as tf
import numpy as np
from keras.backend.tensorflow_backend import set_session
from keras.layers import (Conv1D, Dense, Dropout, Embedding,
                          GlobalMaxPooling1D, Input)
from keras.models import Model
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from tqdm import tqdm

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
set_session(tf.Session(config=config))

# set parameters:
max_features = 500
maxlen = 500
batch_size = 32
embedding_dims = 300
filters = 250
kernel_size = 3
epochs = 10

def transform(input_file):
    print("Reading input file..")

    with open("./" + input_file) as f:
        data = json.load(f)

    abstracts = []
    is_neuro = []

    # Neuroscience MeSH-terms.
    reg = re.compile(r"(D009420|D009457|D009474|D009422|D001520|D011579|D001523|D004191)")

    for article in tqdm(data, desc="Grabbing abstracts and mesh terms"):
        abstracts.append("\n".join([x["text"] for x in article["abstract"]]))
        is_neuro.append(0)
        for mesh_term in article["mesh_list"]:
            if reg.match(mesh_term["mesh_id"]):
                is_neuro[-1] = 1
                break

    del data # Release the huge chunk of memory, ugly. TODO: structure code properly.

    #print("\n",abstracts[0])
    #print(is_neuro[0])

    print("Running vectorizer.")
    vectorizer = CountVectorizer(ngram_range=(1,1))
    abstracts = vectorizer.fit_transform(abstracts)
    print("abstract shape:", abstracts.shape)
    print("abstracts[0]:", abstracts[0][0])

    label_encoder = LabelEncoder()
    is_neuro = label_encoder.fit_transform(is_neuro)
    print("is_neuro shape", is_neuro.shape)
    print("is_neuro label_encoder", is_neuro)
    print("is_neuro class labels", label_encoder.classes_)

    one_hot_encoder = OneHotEncoder(sparse=False)
    is_neuro = one_hot_encoder.fit_transform(is_neuro.reshape(1,-1))
    print("is_neuro 1hot shape", is_neuro.shape)
    print("is_neuro 1hot", is_neuro)

    padding = tf.constant([[0,0],[0,maxlen]])
    abstracts = tf.pad(np.asarray(abstracts), padding)
    abstracts = tf.slice(abstracts, [0,0], [-1, maxlen])

    #abstracts = sequence.pad_sequences(abstracts, padding='post')

    return (abstracts, is_neuro)


def build_model():

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
                class_weight={0: 1.0, 1: 50})
        precision = model_hist.history['val_precision'][0]
        recall = model_hist.history['val_recall'][0]
        f_score = (2.0 * precision * recall) / (precision + recall)
        print("epoch", epoch, "F-score:", f_score)

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
