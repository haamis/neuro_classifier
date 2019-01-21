import pickle, sys
import keras_metrics
import tensorflow as tf
import numpy as np

from keras.backend.tensorflow_backend import set_session
from keras.layers import (Dense, Input)
from keras.models import Model

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder

from nltk.corpus import stopwords

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
set_session(tf.Session(config=config))

# set parameters:
max_features = 10000
batch_size = 1024
filters = 250
kernel_size = 3
epochs = 25

def transform(abstracts_file, is_neuro_file):
    print("Reading input files..")

    abstracts = load_data("./" + abstracts_file)
    is_neuro = load_data("./" + is_neuro_file)

    print("Running vectorizer..")
    vectorizer = TfidfVectorizer(max_features=max_features, binary=True, ngram_range=(1,1), stop_words=stopwords.words("english"))
    abstracts = vectorizer.fit_transform(abstracts)
    print("abstract shape:", abstracts.shape)
    print("abstracts[0]:", abstracts[0][0])

    print("Splitting..")
    abstracts_train, abstracts_test, is_neuro_train, is_neuro_test = train_test_split(abstracts, is_neuro, test_size=0.2)

    one_hot_encoder = OneHotEncoder(sparse=False, categories='auto')
    is_neuro_train = one_hot_encoder.fit_transform(np.asarray(is_neuro_train).reshape(-1,1))
    is_neuro_test = one_hot_encoder.fit_transform(np.asarray(is_neuro_test).reshape(-1,1))
    print("is_neuro 1hot shape", is_neuro_train.shape)
    print("is_neuro 1hot", is_neuro_train)

    return (abstracts_train, is_neuro_train, abstracts_test, is_neuro_test)


def build_model():

    feature_count = abstracts_train.shape[1]

    x = Input(shape=(feature_count,))

    hidden = Dense(200, activation="tanh")(x)

    out = Dense(2, activation='softmax')(hidden)

    model = Model(x, out)

    model.compile(loss='categorical_crossentropy',
                optimizer='adam',
                metrics=[keras_metrics.precision(), keras_metrics.recall()])

    print(model.summary())

    for epoch in range(epochs):

        model_hist = model.fit(abstracts_train, is_neuro_train,
                batch_size=batch_size,
                epochs=1,
                validation_data=(abstracts_test, is_neuro_test))
        precision = model_hist.history['val_precision'][0]
        recall = model_hist.history['val_recall'][0]
        f_score = (2.0 * precision * recall) / (precision + recall)
        print("epoch", epoch + 1, "F-score:", f_score, "\n")

    return model
    

def dump_data(file_name, data):

    with open(file_name, "wb") as f:
        pickle.dump(data, f)


def load_data(file_name):
    
    with open(file_name, "rb") as f:
        return pickle.load(f)


(abstracts_train, is_neuro_train, abstracts_test, is_neuro_test) = transform(sys.argv[1], sys.argv[2])

#abstracts = load_data("../abstracts_dump")
#is_neuro = load_data("../is_neuro_dump")

model = build_model()
#dump_data("../model_dump", model)

#model = load_data("../model_dump")
