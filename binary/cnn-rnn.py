import pickle, sys
import keras_metrics
import tensorflow as tf
import numpy

from keras.backend.tensorflow_backend import set_session
from keras.layers import Conv1D, Dense, Embedding, Bidirectional, GlobalMaxPooling1D, Input, Concatenate
from keras.layers import CuDNNLSTM as LSTM
from keras.layers import CuDNNGRU as GRU
from keras.models import Model
from keras.preprocessing import sequence, text
import keras.utils
from keras.callbacks import EarlyStopping

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

from gensim.models import KeyedVectors

#from nltk.corpus import stopwords

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
set_session(tf.Session(config=config))

# set parameters:
batch_size = 512
filters = 250
kernel_size = 3
epochs = 50

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
        for one_word in one_text:
            if one_word in word_vocab:
                vectorized_text.append(word_vocab[one_word].index) # the .index comes from gensim's vocab
            else:
                vectorized_text.append(1) # OOV
        vectorized_texts.append(numpy.array(vectorized_text))
    vectorized_texts = sequence.pad_sequences(vectorized_texts, padding='post', maxlen=max_len)
    return numpy.array(vectorized_texts)

def transform(abstracts_file, is_neuro_file):

    print("Processing word embeddings..")
    vector_model = KeyedVectors.load_word2vec_format("../../PubMed-and-PMC-w2v.bin", binary=True, limit=100000)

    for word_record in vector_model.vocab.values():
        word_record.index += 2

    word_embeddings = vector_model.vectors 

    two_random_rows = numpy.random.uniform(low=-0.01, high=0.01, size=(2, word_embeddings.shape[1]))
    word_embeddings = numpy.vstack([two_random_rows, word_embeddings])

    word_embeddings = keras.utils.normalize(word_embeddings)

    _, embedding_dims = word_embeddings.shape

    print("Reading input files..")
    abstracts = load_data("./" + abstracts_file)
    is_neuro = load_data("./" + is_neuro_file)

    print("Tokenizing..")
    abstracts = [text.text_to_word_sequence(a) for a in abstracts]
    #stop_words = stopwords.words("english")
    #abstracts = [[word for word in abstract if not word in stop_words] for abstract in abstracts]

    print("Splitting..")
    abstracts_train, abstracts_test, is_neuro_train, is_neuro_test = train_test_split(abstracts, is_neuro, test_size=0.2)
    #del abstracts, is_neuro # Memory management :p

    print(abstracts_train[0])

    print("Vectorizing train set..")
    abstracts_train = vectorize(abstracts_train, vector_model.vocab)
    print("Train set shape:", abstracts_train.shape)
    _, longest_train_sent = abstracts_train.shape
    print("Vectorizing test set..")
    abstracts_test = vectorize(abstracts_test, vector_model.vocab, max_len=longest_train_sent)
    print("Test set shape:", abstracts_test.shape)
    vector_model_length = len(vector_model.vocab)
    #del vector_model # Memory management :p

    one_hot_encoder = OneHotEncoder(sparse=False, categories='auto')
    is_neuro_train = one_hot_encoder.fit_transform(numpy.asarray(is_neuro_train).reshape(-1,1))
    is_neuro_test = one_hot_encoder.fit_transform(numpy.asarray(is_neuro_test).reshape(-1,1))

    _, sequence_len = abstracts_train.shape

    return abstracts_train, abstracts_test, is_neuro_train, is_neuro_test, sequence_len, vector_model_length, embedding_dims, word_embeddings

def build_model(abstracts_train, abstracts_test, is_neuro_train, is_neuro_test, sequence_len, vector_model_length, embedding_dims, word_embeddings):

    print("Building model..")
    input_layer = Input(shape=(sequence_len,))

    embedding_layer = Embedding(vector_model_length+2,
                        embedding_dims, trainable=False,
                        mask_zero=False, weights=[word_embeddings])(input_layer)

    #conv_result = Conv1D(filters, 3, padding='valid', activation='relu', strides=1)(embedding_layer)

    rnn_layer1 = Bidirectional(GRU(10, return_sequences=True))(embedding_layer)

    rnn_layer2 = Bidirectional(GRU(10, return_sequences=True))(rnn_layer1)

    rnn_layer3 = Bidirectional(GRU(10, return_sequences=False))(rnn_layer2)

    #rnn_layer4 = Bidirectional(GRU(10, return_sequences=True))(rnn_layer3)

    #pooled = (GlobalMaxPooling1D())(rnn_layer3)

    #hidden = Dense(200, activation='tanh')(pooled)

    output_layer = Dense(2, activation='softmax')(rnn_layer3)

    model = Model(input_layer, output_layer)

    model.compile(loss='categorical_crossentropy',
                optimizer='adam',
                metrics=[keras_metrics.precision(), keras_metrics.recall(), keras_metrics.f1_score()])

    print(model.summary())

    #es_callback = EarlyStopping(monitor='val_loss', min_delta=0, patience=2, verbose=1, mode='auto')

    model_hist = model.fit(abstracts_train, is_neuro_train,
                            batch_size=batch_size,
                            epochs=epochs,
                            validation_data=(abstracts_test, is_neuro_test))
                            #callbacks=[es_callback])
    
build_model(*transform(sys.argv[1], sys.argv[2]))
