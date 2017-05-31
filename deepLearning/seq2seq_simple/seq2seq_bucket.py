# -*- coding: utf-8 -*-

import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

import tensorflow as tf
import numpy as np

from pprint import pprint

seq_length = 5
batch_size = 64

vocab_size = 7
embedding_dim = 50

memory_dim = 100

''' 
An input to an embedding encoder, for example, 
would be a list of seq_length tensors, 
each of which is of dimension batch_size 
(specifying the embedding indices to input at a particular timestep).

encoder 입력은 tensor의 list이며, encoder의 길이는 sequence의 길이이고, tensor 크기는 batch_size이다 
[
    ['안', '반', '하']
    ['녕', '가', '이']
    ['하', '워', ' ']
    ['세', '요', ' ']
    ['요', ' ', ' ']
]
각 tensor의 길이는 batch_size 이다.
'''
encoder_inputs = [tf.placeholder(tf.int32, shape=(None,), name="input{0}".format(t)) for t in range(seq_length)]
targets = [tf.placeholder(tf.int32, shape=(None,), name="targets{0}".format(t)) for t in range(seq_length)]

weights = [tf.ones_like(labels_t, dtype=tf.float32) for labels_t in targets]

# Decoder input: prepend some "GO" token and drop the final
# token of the encoder input
decoder_inputs = ([tf.zeros_like(encoder_inputs[0], dtype=np.int32, name="GO")] + targets[:-1])

# Initial memory value for recurrence.
prev_mem = tf.zeros((batch_size, memory_dim))


cell = tf.contrib.rnn.core_rnn_cell.GRUCell(memory_dim)

def seq2seq(encoder_inputs, decoder_inputs):
    return tf.contrib.legacy_seq2seq.embedding_rnn_seq2seq(encoder_inputs, decoder_inputs, 
                                        cell, vocab_size, vocab_size, embedding_dim)

def seq2seq_test(encoder_inputs, decoder_inputs):
    return tf.contrib.legacy_seq2seq.embedding_rnn_seq2seq(encoder_inputs, decoder_inputs, 
                                        cell, vocab_size, vocab_size, embedding_dim, feed_previous=True)

buckets = [(8, 10), (12, 14), (16, 19)];
with tf.variable_scope('train_test'):
    outputs, loss = tf.contrib.legacy_seq2seq.model_with_buckets(encoder_inputs, decoder_inputs, targets, weights, buckets, 
                                                seq2seq, softmax_loss_function=None, per_example_loss=False)

with tf.variable_scope("train_test", reuse = True):
    outputs_test, loss_test = tf.contrib.legacy_seq2seq.model_with_buckets(encoder_inputs, decoder_inputs, targets, weights, buckets, 
                                                seq2seq, softmax_loss_function=None, per_example_loss=False)

learning_rate = 0.05
momentum = 0.9
optimizer = tf.train.MomentumOptimizer(learning_rate, momentum)
train_op = optimizer.minimize(loss)


def train_batch(batch_size):

    feed_dict = {encoder_inputs[t]: X[t] for t in range(seq_length)}
    feed_dict.update({targets[t]: Y[t] for t in range(seq_length)})

    _, loss_t = sess.run([train_op, loss], feed_dict)
    
    return loss_t



# start a TF session
sess = tf.Session()
sess.run(tf.global_variables_initializer())

# Train input
X = [np.random.choice(vocab_size, size=(seq_length,), replace=False) for _ in range(batch_size)]
Y = X[:]

# Dimshuffle to seq_len * batch_size
X = np.array(X).T
Y = np.array(Y).T

for t in range(500):
	loss_t = train_batch(X, Y)


'''
Basic autoencoder test
'''
X_batch = [np.random.choice(vocab_size, size=(seq_length,), replace=False) for _ in range(10)]
X_batch = np.array(X_batch).T

feed_dict = {encoder_inputs[t]: X_batch[t] for t in range(seq_length)}
output_batch = sess.run(outputs_test, feed_dict)

print(X_batch)
pprint([logits_t.argmax(axis=1) for logits_t in output_batch]) #axis=0: col, axis=1: row 

