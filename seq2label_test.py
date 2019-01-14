import pickle, sys
import keras_metrics
import tensorflow as tf
import numpy

from keras.backend.tensorflow_backend import set_session
from keras.layers import (Conv1D, Dense, Embedding,
                          GlobalMaxPooling1D, Input, Concatenate)
from keras.models import Model
from keras.preprocessing import sequence, text
import keras.utils

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from imblearn.under_sampling import RandomUnderSampler

from gensim.models import KeyedVectors

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
set_session(tf.Session(config=config))

# set parameters:
batch_size = 320
filters = 250
kernel_size = 3
epochs = 10

def dump_data(file_name, data):

    with open(file_name, "wb") as f:
        pickle.dump(data, f)


def load_data(file_name):

    with open(file_name, "rb") as f:
        return pickle.load(f)


def vectorize(texts, word_vocab, max_len=None):

    vectorized_texts = []
    for one_text in texts:
        vectorized_text = []
        for one_word in text.text_to_word_sequence(one_text):
            if one_word in word_vocab:
                vectorized_text.append(word_vocab[one_word].index) # the .index comes from gensim's vocab
            else:
                vectorized_text.append(1) # OOV
        vectorized_texts.append(numpy.array(vectorized_text))
    print(len(vectorized_texts), len(vectorized_texts[0]))
    vectorized_texts = sequence.pad_sequences(vectorized_texts, padding='post', maxlen=max_len)
    return numpy.array(vectorized_texts)

print("Processing word embeddings..")
vector_model = KeyedVectors.load_word2vec_format("../PubMed-and-PMC-w2v.bin", binary=True, limit=100000)

for word_record in vector_model.vocab.values():
    word_record.index += 2
    
word_embeddings = vector_model.vectors 

two_random_rows = numpy.random.uniform(low=-0.01, high=0.01, size=(2, word_embeddings.shape[1]))
word_embeddings = numpy.vstack([two_random_rows, word_embeddings])

word_embeddings = keras.utils.normalize(word_embeddings)

_, embedding_dims = word_embeddings.shape

print("Reading input files..")
abstracts = load_data("./" + sys.argv[1])
is_neuro = load_data("./" + sys.argv[2])

print("Splitting..")
abstracts_train, abstracts_test, is_neuro_train, is_neuro_test = train_test_split(abstracts, is_neuro, test_size=0.2)
del abstracts, is_neuro # Memory management :p

print("Vectorizing train set..")
abstracts_train = vectorize(abstracts_train, vector_model.vocab)
print("Train set shape:", abstracts_train.shape)
_, longest_train_sent = abstracts_train.shape
print("Vectorizing test set..")
abstracts_test = vectorize(abstracts_test, vector_model.vocab, longest_train_sent)
print("Test set shape:", abstracts_test.shape)
vector_model_length = len(vector_model.vocab)
del vector_model # Memory management :p

#     print("Undersampling..")
#     rus = RandomUnderSampler(random_state=0, sampling_strategy='auto')
#     abstracts_train, is_neuro_train = rus.fit_resample(abstracts_train, is_neuro_train)
#abstracts_test, is_neuro_test = rus.fit_resample(abstracts_test, is_neuro_test)

one_hot_encoder = OneHotEncoder(sparse=False, categories='auto')
is_neuro_train = one_hot_encoder.fit_transform(numpy.asarray(is_neuro_train).reshape(-1,1))
is_neuro_test = one_hot_encoder.fit_transform(numpy.asarray(is_neuro_test).reshape(-1,1))

example_count, sequence_len = abstracts_train.shape

# Building the model.
print("Building model..")
input_layer = Input(shape=(sequence_len,))

embedding_layer = Embedding(vector_model_length+2,
                    embedding_dims,
                    mask_zero=False)

embeddings = embedding_layer(input_layer)

conv_res = []
for width in range(2,5):

    conv_result = Conv1D(filters, width, padding='valid', activation='relu', strides=1)(embeddings)
    pooled = (GlobalMaxPooling1D())(conv_result) 
    conv_res.append(pooled)

concatenated = (Concatenate())(conv_res)

# We add a vanilla hidden layer:
output_layer = Dense(2, activation='softmax')(concatenated)

model = Model(input_layer, output_layer)

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
