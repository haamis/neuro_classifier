import pickle, sys, lzma
import tensorflow as tf
import numpy as np
import keras.backend as K

from scipy.sparse import lil_matrix

from keras.backend.tensorflow_backend import set_session
from keras.layers import Dense, Flatten, Lambda, Dropout
from keras.models import Model
from keras.optimizers import Adam
from keras.utils import multi_gpu_model

from keras_bert.loader import load_trained_model_from_checkpoint
from keras_bert.bert import *
from keras_bert import AdamWarmup, calc_train_steps

from sklearn.metrics import precision_recall_fscore_support

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
set_session(tf.Session(config=config))

# set parameters:
batch_size = 8
gpus = 1
epochs = 15
freeze_bert = True

def load_data(file_name):

    with lzma.open(file_name, "rb") as f:
        return pickle.load(f)

def build_model(abstracts_train, abstracts_test, labels_train, labels_test, sequence_len):

    checkpoint_file = "../../biobert_pubmed/biobert_model.ckpt"
    config_file = "../../biobert_pubmed/bert_config.json"

    biobert = load_trained_model_from_checkpoint(config_file, checkpoint_file, training=False, seq_len=sequence_len)
    #biobert_train = load_trained_model_from_checkpoint(config_file, checkpoint_file, training=True, seq_len=sequence_len)

    # Unfreeze bert layers.
    if not freeze_bert:
        for layer in biobert.layers[:]:
            layer.trainable = True

    print(biobert.input)
    print(biobert.layers[-1].output)

    print(tf.slice(biobert.layers[-1].output, [0, 0, 0], [-1, 1, -1]))

    slice_layer = Lambda(lambda x: tf.slice(x, [0, 0, 0], [-1, 1, -1]))(biobert.layers[-1].output)

    flatten_layer = Flatten()(slice_layer)

    dropout_layer = Dropout(0.1)(flatten_layer)

    output_layer = Dense(labels_train.shape[1], activation='sigmoid')(dropout_layer)

    if gpus > 1:
        base_model = Model(biobert.input, output_layer)
        model = multi_gpu_model(base_model, gpus=gpus, cpu_merge=True, cpu_relocation=False)
    else:
        model = Model(biobert.input, output_layer)

    print(model.summary(line_length=118))

    print("Number of GPUs in use:", gpus)

    learning_rate = 0.001

    total_steps, warmup_steps = calc_train_steps(
        num_example=abstracts_train.shape[0],
        batch_size=batch_size,
        epochs=epochs,
        warmup_proportion=0.1)

    model.compile(loss='binary_crossentropy',
                optimizer=AdamWarmup(total_steps, warmup_steps, lr=learning_rate, min_lr=1e-5))#, decay=0.01))
    
    ### for no pretrain
    # Unfreezing bert before saving, for debug.
    for layer in biobert.layers[:]:
        layer.trainable = True
    
    if gpus > 1:
        base_model.save(sys.argv[5])
    else:
        model.save(sys.argv[5])

    sys.exit()
    ### /for no pretrain

    best_f1 = 0.0
    stale_epochs = 0

    for epoch in range(epochs):
        print("Epoch", epoch + 1)
        print("batch size:", batch_size)
        print("learning rate:", K.eval(model.optimizer.lr))
        model.fit([abstracts_train, lil_matrix(abstracts_train.shape)], labels_train,
            batch_size=batch_size*gpus,
            epochs=1,
            validation_data=[[abstracts_test, lil_matrix(abstracts_test.shape)], labels_test])
        print("Predicting probabilities..")
        labels_prob = model.predict([abstracts_test, lil_matrix(abstracts_test.shape)])

        print("Probabilities to labels..")
        labels_pred = lil_matrix(labels_prob.shape, dtype='b')
        labels_pred[labels_prob>0.5] = 1

        precision, recall, f1, _ = precision_recall_fscore_support(labels_test, labels_pred, average='micro')
        print("Precision:", precision)
        print("Recall:", recall)
        print("F1-score:", f1, "\n")
        
        if f1 > best_f1:
            best_f1 = f1
            stale_epochs = 0
            print("Saving model..\n")

            # Unfreezing bert before saving, for debug.
            for layer in biobert.layers[:]:
                layer.trainable = True
            
            if gpus > 1:
                base_model.save(sys.argv[5])
            else:
                model.save(sys.argv[5])

            # Freeze it back for training if necessary.
            if freeze_bert:
                for layer in biobert.layers[:]:
                    layer.trainable = False
        else:
            stale_epochs += 1
            if stale_epochs >= 4:
                break

if __name__ == '__main__':

    print("Reading input files..")

    abstracts_train = load_data(sys.argv[1])
    abstracts_test = load_data(sys.argv[2])
    labels_train = load_data(sys.argv[3])
    labels_test = load_data(sys.argv[4])

    _, sequence_len = abstracts_train.shape

    build_model(abstracts_train, abstracts_test, labels_train, labels_test, sequence_len)

