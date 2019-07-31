import pickle, sys, lzma
import tensorflow as tf
import numpy as np
import keras.backend as K
import horovod.keras as hvd
import math

from scipy.sparse import lil_matrix

from keras.callbacks import Callback, EarlyStopping, ModelCheckpoint
from keras.layers import Dense, Flatten, Lambda, Dropout, Multiply, Permute, RepeatVector
from keras.models import Model

from keras_bert.loader import load_trained_model_from_checkpoint
from keras_bert import AdamWarmup, calc_train_steps

from sklearn.metrics import precision_recall_fscore_support

hvd.init()

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.visible_device_list = str(hvd.local_rank())
K.set_session(tf.Session(config=config))

# set parameters:
batch_size = 1
learning_rate = 4e-5
epochs = int(math.ceil( 20 // hvd.size() ))

class Metrics(Callback):

    def on_epoch_end(self, batch, logs={}):
        print("Predicting probabilities..")
        labels_prob = self.model.predict([abstracts_test, lil_matrix(abstracts_test.shape)])

        print("Probabilities to labels..")
        labels_pred = lil_matrix(labels_prob.shape, dtype='b')
        labels_pred[labels_prob>0.5] = 1

        precision, recall, f1, _ = precision_recall_fscore_support(labels_test, labels_pred, average="micro")
        print("Precision:", precision)
        print("Recall:", recall)
        print("F1-score:", f1, "\n")

def load_data(file_name):

    with lzma.open(file_name, "rb") as f:
        return pickle.load(f)

def build_model(abstracts_train, abstracts_test, labels_train, labels_test, sequence_len):

    checkpoint_file = "../../biobert_pubmed/biobert_model.ckpt"
    config_file = "../../biobert_pubmed/bert_config.json"

    biobert = load_trained_model_from_checkpoint(config_file, checkpoint_file,
                                                training=False, trainable=True,
                                                seq_len=sequence_len)

    #slice_layer = Lambda(lambda x: tf.slice(x, [0, 0, 0], [-1, 1, -1]))(biobert.layers[-1].output)

    #flatten_layer = Flatten()(slice_layer)

    #dropout_layer = Dropout(0.1)(flatten_layer)

    attention = Dense(1, activation="softmax", name="attention")(biobert.layers[-1].output)

    # Hackish way to drop masking, flatten won't work otherwise.
    attention = Lambda(lambda x: x, name="drop_mask_1")(attention)

    attention = Flatten()(attention)

    attention = RepeatVector(768)(attention)
    
    attention = Permute((2,1))(attention)

    attention = Multiply()([attention, biobert.layers[-1].output])

    # Hackish way to drop masking, flatten won't work otherwise.
    attention = Lambda(lambda x: x, name="drop_mask_2")(attention)

    attention = Flatten()(attention)
    
    output_layer = Dense(labels_train.shape[1], activation="sigmoid", name="decision")(attention)

    model = Model(biobert.input, output_layer)

    metrics = Metrics()
    early_stopping = EarlyStopping(patience=10, verbose=1)

    total_steps, warmup_steps = calc_train_steps(num_example=abstracts_train.shape[0],
                                                batch_size=batch_size, epochs=epochs,
                                                warmup_proportion=0.1)

    optimizer = AdamWarmup(total_steps, warmup_steps, lr=learning_rate)
    optimizer = hvd.DistributedOptimizer(optimizer)

    print("Compiling model")

    model.compile(loss="binary_crossentropy", optimizer=optimizer)

    callbacks = [hvd.callbacks.BroadcastGlobalVariablesCallback(0), metrics, early_stopping]
    
    if hvd.rank() == 0:
        callbacks.append(ModelCheckpoint(sys.argv[5][:-3] + "-checkpoint-{epoch}.h5"))
        print(model.summary(line_length=118))
        print("Number of GPUs in use:", hvd.size())
        print("Batch size:", batch_size)
        print("Learning rate:", learning_rate)

    model.fit([abstracts_train, lil_matrix(abstracts_train.shape)], labels_train,
        batch_size=batch_size, epochs=epochs, callbacks=callbacks,
        validation_data=[[abstracts_test, lil_matrix(abstracts_test.shape)], labels_test])
    if hvd.rank() == 0:
        model.save(sys.argv[5])

if __name__ == "__main__":

    print("Reading input files..")

    abstracts_train = load_data(sys.argv[1])
    print(abstracts_train.shape)
    abstracts_train = abstracts_train[:,0:16]
    print(abstracts_train.shape)
    abstracts_test = load_data(sys.argv[2])
    abstracts_test = abstracts_test[:,0:16]
    labels_train = load_data(sys.argv[3])
    labels_test = load_data(sys.argv[4])

    _, sequence_len = abstracts_train.shape

    build_model(abstracts_train, abstracts_test, labels_train, labels_test, sequence_len)
