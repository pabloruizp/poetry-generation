import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import numpy as np
import re
import sys
from tensorflow.keras.layers.experimental import preprocessing

vocab = ['\n', ' ', '!', '"', '&', "'", '(', ')', '*', '+', ',', '-', '.', '/', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', ':', ';', '=', '?', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', '[', ']', '_', '`', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', '\xa0', '¡', 'ª', 'º', '¿', 'À', 'Á', 'Ã', 'Ç', 'É', 'Í', 'Ñ', 'Ó', 'Õ', 'Ú', 'à', 'á', 'ã', 'ä', 'ç', 'è', 'é', 'ê', 'í', 'î', 'ï', 'ñ', 'ó', 'ô', 'õ', 'ö', 'ù', 'ú', 'ü', 'ý']

char2id = preprocessing.StringLookup(vocabulary=list(vocab))
id2char = preprocessing.StringLookup(vocabulary=char2id.get_vocabulary(), invert=True)

vocab_size = len(vocab)
embedding_dim = 256
rnn_units = 1024

class PoetryGeneratorModel(tf.keras.Model):
  def __init__(self, vocab_size, embeding_dim, rnn_units):
    super().__init__(self)
    self.embedding = tf.keras.layers.Embedding(vocab_size, embeding_dim)
    self.gru = tf.keras.layers.GRU(rnn_units, return_sequences=True, return_state=True)
    self.dense = tf.keras.layers.Dense(vocab_size)

  def call(self, inputs, states=None, return_state=False, training=False):
    x = inputs
    x = self.embedding(x, training=training)
    if states is None:
      states = self.gru.get_initial_state(x)
    x, states = self.gru(x, initial_state=states, training=training)
    x = self.dense(x, training=training)

    if return_state:
      return x, states
    else:
      return x

model = PoetryGeneratorModel(vocab_size=len(char2id.get_vocabulary()), embeding_dim=embedding_dim,rnn_units=rnn_units)
model.compile(optimizer='adam', loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])

initial = ['S']
initial = [char for char in initial]
initialIDs = char2id(initial)
initialIDs = tf.expand_dims(initialIDs, axis=0)
model(initialIDs, states=None, return_state=False)

model.load_weights('model/poeta.h5')

def generateText(model, nChars, initialString):
  states = None
  initial = [initialString]
  poem = initial[0]

  for i in range(nChars):
    initial = [char for char in initial]
    initialIDs = char2id(initial)
    initialIDs = tf.expand_dims(initialIDs, axis=0)
    pred, states = model(initialIDs, states=states, return_state=True)
    pred = pred[:, -1, :]
    pred = tf.random.categorical(pred, num_samples=1)
    pred = tf.squeeze(pred, axis=-1)
    initial = id2char(pred)
    poem += id2char(pred)[0].numpy().decode('utf-8')

  poem = poem.split('\n')
  poem = '\n'.join([line.strip() for line in poem])
  poem = re.sub(' +', ' ', poem)
  print('_'*50)
  print(poem, '\n')

generateText(model, 500, sys.argv[1])
