import pickle, sys
import keras_metrics
import tensorflow as tf
import numpy as np

from keras.backend.tensorflow_backend import set_session
from keras.layers import (Conv1D, Dense, Dropout, Embedding,
                          GlobalMaxPooling1D, Input)
from keras.models import Model
from keras.preprocessing import sequence

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from imblearn.under_sampling import RandomUnderSampler

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
set_session(tf.Session(config=config))

# set parameters:
max_features = 500
maxlen = 500
batch_size = 192
embedding_dims = 300
filters = 250
kernel_size = 3
epochs = 50

def transform(abstracts_file, is_neuro_file):
    print("Reading input files..")

    abstracts = load_data("./" + abstracts_file)
    is_neuro = load_data("./" + is_neuro_file)

    #print("\n",abstracts[0])
    #print(is_neuro[0])

    print("Running vectorizer..")
    vectorizer = CountVectorizer(max_features=max_features, ngram_range=(1,1))
    abstracts = vectorizer.fit_transform(abstracts)
    print("abstract shape:", abstracts.shape)
    print("abstracts[0]:", abstracts[0][0])

    print("Splitting..")
    abstracts_train, abstracts_test, is_neuro_train, is_neuro_test = train_test_split(abstracts, is_neuro, test_size=0.2)

    print("Undersampling..")
    rus = RandomUnderSampler(random_state=0, sampling_strategy='auto')
    abstracts_train, is_neuro_train = rus.fit_resample(abstracts_train, is_neuro_train)
    #abstracts_test, is_neuro_test = rus.fit_resample(abstracts_test, is_neuro_test)
    
    label_encoder = LabelEncoder()
    is_neuro_train = label_encoder.fit_transform(is_neuro_train)
    is_neuro_test = label_encoder.fit_transform(is_neuro_test)
    print("is_neuro shape", is_neuro_train.shape)
    print("is_neuro label_encoder", is_neuro_train)
    print("is_neuro class labels", label_encoder.classes_)

    one_hot_encoder = OneHotEncoder(sparse=False)
    is_neuro_train = one_hot_encoder.fit_transform(is_neuro_train.reshape(-1,1))
    is_neuro_test = one_hot_encoder.fit_transform(is_neuro_test.reshape(-1,1))
    print("is_neuro 1hot shape", is_neuro_train.shape)
    print("is_neuro 1hot", is_neuro_train)
    
    
    
    #abstracts = sequence.pad_sequences(abstracts.todense(abstracts), padding='post')

    return (abstracts_train, is_neuro_train, abstracts_test, is_neuro_test)


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

    model.compile(loss='categorical_crossentropy',
                optimizer='adam',
                metrics=[keras_metrics.precision(), keras_metrics.recall()])

    print(model.summary())

    for epoch in range(epochs):

        model_hist = model.fit(abstracts_train, is_neuro_train,
                batch_size=batch_size,
                epochs=1,
                validation_data = (abstracts_test, is_neuro_test))
        precision = model_hist.history['val_precision'][0]
        recall = model_hist.history['val_recall'][0]
        f_score = (2.0 * precision * recall) / (precision + recall)
        print("epoch", epoch, "F-score:", f_score, "\n")

    return model


def dump_data(file_name, dump_data):

    with open(file_name, "wb") as f:
        pickle.dump(dump_data, f)


def load_data(file_name):
    
    with open(file_name, "rb") as f:
        return pickle.load(f)


(abstracts_train, is_neuro_train, abstracts_test, is_neuro_test) = transform(*sys.argv[1:])

#abstracts = load_data("../abstracts_dump")
#is_neuro = load_data("../is_neuro_dump")

model = build_model()
#dump_data("../model_dump", model)

#model = load_data("../model_dump")
