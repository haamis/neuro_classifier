import pickle, sys
import tensorflow as tf
import numpy as np

from timeit import default_timer as timer

from scipy.sparse import csr_matrix, lil_matrix

from keras.backend.tensorflow_backend import set_session
from keras.layers import Bidirectional, Concatenate, Conv1D, Dense, Dropout, Input, Embedding, GlobalMaxPooling1D
from keras.models import Model
from keras.preprocessing import sequence, text
from keras.utils import normalize
from keras.optimizers import Adam
from keras.layers import CuDNNGRU as GRU

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import precision_score, recall_score, f1_score, precision_recall_fscore_support

from gensim.models import KeyedVectors

from tqdm import tqdm

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
set_session(tf.Session(config=config))

# set parameters:
batch_size = 64
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
    for one_text in tqdm(texts, desc="Vectorizing"):
        vectorized_text = []
        for one_word in one_text:
            if one_word in word_vocab:
                vectorized_text.append(word_vocab[one_word].index) # the .index comes from gensim's vocab
            else:
                vectorized_text.append(1) # OOV
        vectorized_texts.append(np.array(vectorized_text, dtype='uint32'))
    vectorized_texts = sequence.pad_sequences(vectorized_texts, padding='post', maxlen=max_len)
    return np.array(vectorized_texts)

def transform(abstracts_file, mesh_file):
    print("Processing word embeddings..")
    vector_model = KeyedVectors.load_word2vec_format("../../PubMed-and-PMC-w2v.bin", binary=True, limit=1000000)

    for word_record in vector_model.vocab.values():
        word_record.index += 2
        
    word_embeddings = vector_model.vectors 

    two_random_rows = np.random.uniform(low=-0.01, high=0.01, size=(2, word_embeddings.shape[1]))
    word_embeddings = np.vstack([two_random_rows, word_embeddings])

    word_embeddings = normalize(word_embeddings)

    _, embedding_dims = word_embeddings.shape

    print("Reading input files..")

    abstracts = load_data("./" + abstracts_file)
    labels = load_data("./" + mesh_file)

    print("Tokenizing..")
    abstracts = [text.text_to_word_sequence(a) for a in abstracts]

    print("Vectorizing abstracts..")
    abstracts = vectorize(abstracts, vector_model.vocab, max_len=400)
    print("Abstracts shape:", abstracts.shape)
    #print(np.dtype(abstracts[0]))
    vector_model_length = len(vector_model.vocab)

    print("Binarizing labels..")
    mlb = MultiLabelBinarizer(sparse_output=True)
    labels = mlb.fit_transform(labels)
    labels = labels.astype('b')
    print("Labels shape:", labels.shape)
    print(np.dtype(labels))

    print("Splitting..")
    abstracts_train, abstracts_test, labels_train, labels_test = train_test_split(abstracts, labels, test_size=0.1)

    _, sequence_len = abstracts_train.shape

    return abstracts_train, abstracts_test, labels_train, labels_test, sequence_len, vector_model_length, embedding_dims, word_embeddings


def build_model(abstracts_train, abstracts_test, labels_train, labels_test, sequence_len, vector_model_length, embedding_dims, word_embeddings):

    input_layer = Input(shape=(sequence_len,))

    embedding_layer = Embedding(vector_model_length+2,
                        embedding_dims, trainable=False,
                        mask_zero=False, weights=[word_embeddings])(input_layer)
    del word_embeddings

    dropout_layer = Dropout(0.2)(embedding_layer)

    #rnn_layer1 = Bidirectional(GRU(10, return_sequences=False))(dropout_layer)

    #conv_result = Conv1D(filters, 3, padding='valid', activation='relu', strides=1)(dropout_layer)
    #pooled = GlobalMaxPooling1D()(conv_result)

    conv_res = []
    for width in range(2,5):

        conv_result = Conv1D(filters, width, padding='valid', activation='relu', strides=1)(dropout_layer)
        pooled = (GlobalMaxPooling1D())(conv_result) 
        conv_res.append(pooled)

    concatenated = (Concatenate())(conv_res)
    
    output_layer = Dense(labels_train.shape[1], activation='sigmoid')(concatenated)

    model = Model(input_layer, output_layer)

    model.compile(loss='binary_crossentropy',
                optimizer=Adam(lr=0.001))

    print(model.summary())

    for epoch in range(epochs):

        model.fit(abstracts_train, labels_train,
            batch_size=batch_size,
            epochs=1,
            validation_data=[abstracts_test, labels_test])
        print("Predicting probabilities..")
        labels_prob = model.predict(abstracts_test)
        # label_indices = [np.argpartition(array, -15)[-15:] for array in labels_prob]

        # labels_pred = []
        # for i, array in tqdm(enumerate(labels_prob), desc="Probs to labels"):
        #     temp_array = np.zeros(len(array))
        #     for j in label_indices[i]:     
        #         temp_array[j] = 1
        #     labels_pred.append(temp_array)
        # labels_pred = np.array(labels_pred)
        print("Probabilities to labels..")
        labels_pred = lil_matrix(labels_prob.shape, dtype='b')
        labels_pred[labels_prob>0.5] = 1
        false_positives = lil_matrix(labels_prob.shape)
        false_positives[labels_pred.nonzero()] = 1
        false_positives[labels_test.nonzero()] = 0
        print(false_positives)

        # false_positives = [[1 if (labels_pred[j][i] == 1) and (labels_test[j][i] == 0) else 0 for i in range(labels_pred.shape[1])] for j in range(labels_pred.shape[0])]
        import pdb
        pdb.set_trace()
        print("Epoch", epoch + 1)
        precision, recall, f1, _ = precision_recall_fscore_support(labels_test, labels_pred, average='micro')
        print("Precision:", precision)
        print("Recall:", recall)
        print("F1-score:", f1, "\n")


build_model(*transform(sys.argv[1], sys.argv[2]))

#abstracts = load_data("../abstracts_dump")
#is_neuro = load_data("../is_neuro_dump")

#model = build_model()
#dump_data("../model_dump", model)

#model = load_data("../model_dump")
