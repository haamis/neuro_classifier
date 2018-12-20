import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
from keras.preprocessing import sequence
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Activation
from keras.layers import Embedding, Input
from keras.layers import Conv1D, GlobalMaxPooling1D

from sklearn.preprocessing import OneHotEncoder
from keras.datasets import imdb

import sys
import json
import re
from tqdm import tqdm

config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.3
set_session(tf.Session(config=config))

# set parameters:
max_features = 5000
maxlen = 400
batch_size = 32
embedding_dims = 300
filters = 250
kernel_size = 3
epochs = 2

print("Reading input file..")

with open("./" + sys.argv[1]) as f:
    data = json.load(f)

abstracts = []
is_neuro = []

# Neuroscience MeSH-terms.
reg = re.compile(r"(A08*|A11.650*|A08.637*|A11.671*|A08.675*|C10*|F01*|F02*|F03*|F04*)")

for article in tqdm(data, desc="Grabbing abstracts and mesh terms"):
    abstract_parts = []
    abstracts.append("\n".join([x["text"] for x in article["abstract"]]))
    is_neuro.append(0)
    for mesh_term in article["mesh_list"]:
        if reg.match(mesh_term["mesh_id"]):
            is_neuro[-1] = 1
            break

del data # Release the huge chunk of memory, ugly. TODO: structure code properly.

print(abstracts[0])
print(is_neuro[0])

#input("check ram")

print('Loading data...')
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)
print(len(x_train), 'train sequences')
print(len(x_test), 'test sequences')

print (x_train.shape)

#Since we are using output the size of 2, we will have to do one-hot encoding
#x_test = to_categorical(x_test)
#y_test = to_categorical(y_test)


print('Pad sequences (samples x time)')
x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
x_test = sequence.pad_sequences(x_test, maxlen=maxlen)
print('x_train shape:', x_train.shape)
print('x_test shape:', x_test.shape)

onehot_encoder = OneHotEncoder(sparse=False)
y_test = onehot_encoder.fit_transform(y_test.reshape(-1, 1))
y_train = onehot_encoder.transform(y_train.reshape(-1, 1))



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