import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
from keras.preprocessing import sequence
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Activation
from keras.layers import Embedding, Input
from keras.layers import Conv1D, GlobalMaxPooling1D

from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer

import sys
import json
import re
import pickle
from tqdm import tqdm

""" config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.3
set_session(tf.Session(config=config)) """

# set parameters:
max_features = 5000
maxlen = 400
batch_size = 32
embedding_dims = 300
filters = 250
kernel_size = 3
epochs = 2

def transform_and_save(input_file):
    print("Reading input file..")

    with open("./" + input_file) as f:
        data = json.load(f)

    abstracts = []
    is_neuro = []

    # Neuroscience MeSH-terms.
    reg = re.compile(r"(A08*|A11.650*|A08.637*|A11.671*|A08.675*|C10*|F01*|F02*|F03*|F04*)")

    for article in tqdm(data, desc="Grabbing abstracts and mesh terms"):
        abstract_parts = []
        abstracts.append("\n".join([x["text"] for x in article["abstract"]]))
        is_neuro.append(False)
        for mesh_term in article["mesh_list"]:
            if reg.match(mesh_term["mesh_id"]):
                is_neuro[-1] = True
                break

    del data # Release the huge chunk of memory, ugly. TODO: structure code properly.

    print("\n",abstracts[0])
    print(is_neuro[0])

    #input("check ram")

    vectorizer = CountVectorizer(max_features=100000, binary=True, ngram_range=(1,1))
    abstracts = vectorizer.fit_transform(abstracts)
    print("abstract shape:", abstracts.shape)

    label_encoder = LabelEncoder()
    is_neuro = label_encoder.fit_transform(is_neuro)
    print("is_neuro shape", is_neuro.shape)

    one_hot_encoder = OneHotEncoder(sparse=False)
    is_neuro = one_hot_encoder.fit_transform(is_neuro.reshape(-1,1))
    print("is_neuro 1hot", is_neuro.shape)

    with open("../abstracts_dump", "w") as f:
        pickle.dump(abstracts, f)
    
    with open("../is_neuro_dump", "w") as f:
        pickle.dump(is_neuro, f)

transform_and_save(sys.argv[1])
sys.exit()

print('Build model...')

#Let's define the inputs
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
              metrics=['accuracy'])


model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          validation_data=(x_test, y_test))