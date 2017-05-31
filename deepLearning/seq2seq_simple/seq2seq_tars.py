# -*- coding: utf-8 -*-

import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

import tensorflow as tf
import numpy as np
import sys

from pprint import pprint

ckpt_path = 'ckpt/'
model_name = 'seq2seq_model'

seq_length = 5
batch_size = 64

vocab_size = 7
embedding_dim = 50

memory_dim = 100
num_layers = 1

epochs = 500

''' 
An input to an embedding encoder, for example, 
would be a list of seq_length tensors, 
each of which is of dimension batch_size 
(specifying the embedding indices to input at a particular timestep).

encoder 입력은 sequence 길이를 가진 tensor의 list이며, 
각 tensor의 길이는 batch_size 이다.
'''
encoder_inputs = [tf.placeholder(tf.int32, shape=(None,), name="input{0}".format(t)) for t in range(seq_length)]
labels = [tf.placeholder(tf.int32, shape=(None,), name="labels{0}".format(t)) for t in range(seq_length)]

weights = [tf.ones_like(labels_t, dtype=tf.float32) for labels_t in labels]

# Decoder input: prepend some "GO" token and drop the final
# token of the encoder input
decoder_inputs = ([tf.zeros_like(encoder_inputs[0], dtype=np.int32, name="GO")] + labels[:-1])

# Initial memory value for recurrence.
prev_mem = tf.zeros((batch_size, memory_dim))


cell = tf.contrib.rnn.core_rnn_cell.GRUCell(memory_dim)
stacked_cell = tf.contrib.rnn.core_rnn_cell.MultiRNNCell([cell]*num_layers, state_is_tuple=True)
with tf.variable_scope('train_test'):
    dec_outputs, dec_memory = tf.contrib.legacy_seq2seq.embedding_rnn_seq2seq(
        encoder_inputs, decoder_inputs, stacked_cell, vocab_size, vocab_size, embedding_dim)
	
with tf.variable_scope("train_test", reuse = True):
    dec_outputs_test, dec_memory_test = tf.contrib.legacy_seq2seq.embedding_rnn_seq2seq(
        encoder_inputs, decoder_inputs, stacked_cell, vocab_size, vocab_size, embedding_dim, feed_previous=True)

loss = tf.contrib.legacy_seq2seq.sequence_loss(dec_outputs, labels, weights, vocab_size)


learning_rate = 0.05
momentum = 0.9
optimizer = tf.train.MomentumOptimizer(learning_rate, momentum)
train_op = optimizer.minimize(loss)

def train_batch(X,Y):
    feed_dict = {encoder_inputs[t]: X[t] for t in range(seq_length)}
    feed_dict.update({labels[t]: Y[t] for t in range(seq_length)})

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

# we need to save the model periodically
saver = tf.train.Saver()
sys.stdout.write('\n<log> Training started </log>\n')

# Train
for t in range(epochs):
    try:
        loss_t = train_batch(X, Y)
        if t and t % (epochs//5) == 0:
            # save model to disk
            saver.save(sess, ckpt_path + model_name + '.ckpt', global_step=t)
            print('\nModel saved to disk at iteration #{}'.format(t))
            sys.stdout.flush()
    except KeyboardInterrupt:
        print('Interrupted by user at iteration {}'.format(t))
        break;

'''
Basic autoencoder test
'''
X_batch = [np.random.choice(vocab_size, size=(seq_length,), replace=False) for _ in range(10)]
X_batch = np.array(X_batch).T

feed_dict = {encoder_inputs[t]: X_batch[t] for t in range(seq_length)}
dec_outputs_batch = sess.run(dec_outputs_test, feed_dict)

print(X_batch)
pprint([logits_t.argmax(axis=1) for logits_t in dec_outputs_batch]) #axis=0: col, axis=1: row 

