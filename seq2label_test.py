import pickle, sys
import keras_metrics
import tensorflow as tf
import numpy

from keras.backend.tensorflow_backend import set_session
from keras.layers import (Conv1D, Dense, Embedding,
                          GlobalMaxPooling1D, Input)
from keras.models import Model
from keras.preprocessing import sequence
import keras.utils

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from imblearn.under_sampling import RandomUnderSampler

from gensim.models import KeyedVectors

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
set_session(tf.Session(config=config))

# set parameters:
batch_size = 192
filters = 250
kernel_size = 3
epochs = 10

def dump_data(file_name, data):

    with open(file_name, "wb") as f:
        pickle.dump(data, f)


def load_data(file_name):

    with open(file_name, "rb") as f:
        return pickle.load(f)


def vectorize(texts, word_vocab):

    vectorized_texts = [] # List of sentences, each sentence is a list of words, and each word is a list of features
    for one_text in texts:
        vectorized_text = [] # One sentence, ie list of words, each being a list of features
        for one_word in one_text:
            if one_word in word_vocab:
                vectorized_text.append(word_vocab[one_word].index) # the .index comes from gensim's vocab
            else:
                vectorized_text.append(1) # OOV
        vectorized_texts.append(vectorized_text)

    return numpy.array(vectorized_texts)


print("Processing word embeddings..")
vector_model = KeyedVectors.load_word2vec_format("../PubMed-and-PMC-w2v.bin", binary=True, limit=100000)

for word_record in vector_model.vocab.values():
    word_record.index += 2
    
word_embeddings = vector_model.vectors 

two_random_rows = numpy.random.uniform(low=-0.01, high=0.01, size=(2, word_embeddings.shape[1]))
word_embeddings = numpy.vstack([two_random_rows, word_embeddings])

word_embeddings = keras.utils.normalize(word_embeddings)
print("word_embeddings size:", sys.getsizeof(word_embeddings))

_, embedding_dims = word_embeddings.shape

print("Reading input files..")
abstracts = load_data("./" + sys.argv[1])
is_neuro = load_data("./" + sys.argv[2])

print("Splitting..")
abstracts_train, abstracts_test, is_neuro_train, is_neuro_test = train_test_split(abstracts, is_neuro, test_size=0.2)
print("abstracts size:", sys.getsizeof(abstracts))
print("is_neuro size:", sys.getsizeof(is_neuro))
del abstracts, is_neuro # Memory management :p

print("Vectorizing train set..")
abstracts_train = vectorize(abstracts_train, vector_model.vocab)
print("Vectorizing test set..")
abstracts_test = vectorize(abstracts_test, vector_model.vocab)
vector_model_length = len(vector_model.vocab)
print("vector_model size:", sys.getsizeof(vector_model))
del vector_model # Memory management :p

#     print("Undersampling..")
#     rus = RandomUnderSampler(random_state=0, sampling_strategy='auto')
#     abstracts_train, is_neuro_train = rus.fit_resample(abstracts_train, is_neuro_train)
#abstracts_test, is_neuro_test = rus.fit_resample(abstracts_test, is_neuro_test)

one_hot_encoder = OneHotEncoder(sparse=False, categories='auto')
is_neuro_train = one_hot_encoder.fit_transform(numpy.asarray(is_neuro_train).reshape(-1,1))
is_neuro_test = one_hot_encoder.fit_transform(numpy.asarray(is_neuro_test).reshape(-1,1))

print("Padding..")
abstracts_train = sequence.pad_sequences(abstracts_train, padding='post')
_, longest_train_sent = abstracts_train.shape
abstracts_test = sequence.pad_sequences(abstracts_test, padding='post', maxlen=longest_train_sent)

print("Creating masks..")
sentence_mask_train = numpy.where((abstracts_train > 0,1))
sentence_mask_test = numpy.where((abstracts_test > 0,1))

example_count, sequence_len = abstracts_train.shape

# Building the model.
print("Building model..")
input_layer = Input(shape=(sequence_len,))

embedding_layer = Embedding(vector_model_length+2,
                    embedding_dims,
                    mask_zero=False)

embeddings = embedding_layer(input_layer)

conv_layer = Conv1D(filters, kernel_size, padding='valid', activation='relu', strides=1)
conv_result = conv_layer(embeddings)
pooled = (GlobalMaxPooling1D())(conv_result) 

# We add a vanilla hidden layer:
output_layer = Dense(2, activation='softmax')(pooled)

model = Model(input_layer, output_layer)

model.compile(loss='categorical_crossentropy',
            optimizer='sgd',
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
