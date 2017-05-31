# -*- coding: utf-8 -*-

import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

import tensorflow as tf
import numpy as np
import sys


class Seq2Seq(object):

    def __init__(self, xseq_len, yseq_len, 
            xvocab_size, yvocab_size,
            ckpt_path,
            embedding_dim,
            num_layers, 
            lr=0.0001, 
            epochs=100000, model_name='seq2seq_model'):

        # attach these arguments to self
        self.xseq_len = xseq_len
        self.yseq_len = yseq_len
        self.xvocab_size = xvocab_size
        self.yvocab_size = yvocab_size
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers
        self.ckpt_path = ckpt_path
        self.lr = lr
        self.epochs = epochs
        self.model_name = model_name

        # build comput graph
        sys.stdout.write('<log> Building Graph ')
        self.__graph__()
        sys.stdout.write('</log>')


    # build graph
    def __graph__(self):

        # reset graph
        tf.reset_default_graph()

        ''' 
        An input to an embedding encoder, for example, 
        would be a list of seq_length tensors, 
        each of which is of dimension batch_size 
        (specifying the embedding indices to input at a particular timestep).
        '''

        # encoder inputs : 각 placeholder에는 문장에서 같은 index를 갖는 단어들이 들어간다.
        ''' List of 2D Tensors of shape [batch_size x num_encoder_symbols].
        [
            ['안[0, 1, 0, 0]', '반[0, 0, 1, 0]', '하[0, 0, 0, 1]'], - > [batch_size, num_encoder_symbols]
            ['녕[0, 0, 1, 0]', '가[0, 0, 0, 1]', '이[1, 0, 0, 0]'],
            ['하[1, 0, 0, 0]', '워[0, 1, 0, 0]', ' '],
            ['세[0, 0, 0, 1]', '요[1, 0, 0, 0]', ' '],
            ['요[0, 1, 0, 0]', ' ', ' ']
        ]
        ''' 
        self.encoder_inputs = [ tf.placeholder(shape=[None,], dtype=tf.int64, name='input_{}'.format(t)) for t in range(self.xseq_len) ]

        # labels that represent the real outputs
        '''  List of 1D batch-sized int32 Tensors of the same length as logits.
        [
            [1, 3, 4],
            [2, 3, 0],
            [0, 1, x],
            [3, 0, x],
            [1, x, x]
        ]
        '''
        self.labels = [ tf.placeholder(shape=[None,], dtype=tf.int64, name='label_{}'.format(t)) for t in range(self.yseq_len) ]

        # decoder inputs : 'GO' + [ y1, y2, ... y_t-1 ]
        self.decoder_inputs = [ tf.zeros_like(self.encoder_inputs[0], dtype=tf.int64, name='GO') ] + self.labels[:-1]

        # basic LSTM cell wrapped in Dropout Wrapper
        self.keep_prob = tf.placeholder(tf.float32)
        
        # define the basic cell & dropout cell
        basic_cell = tf.contrib.rnn.core_rnn_cell.BasicLSTMCell(self.embedding_dim, state_is_tuple=True)
        cell = tf.contrib.rnn.core_rnn_cell.DropoutWrapper(basic_cell, output_keep_prob=self.keep_prob)

        # stack cells together : n layered model
        stacked_lstm = tf.contrib.rnn.core_rnn_cell.MultiRNNCell([cell]*self.num_layers, state_is_tuple=True)

        with tf.variable_scope('train_test'):
            self.decoder_outputs, self.decoder_states = tf.contrib.legacy_seq2seq.embedding_rnn_seq2seq(
                self.encoder_inputs, self.decoder_inputs, stacked_lstm, self.xvocab_size, self.yvocab_size, self.embedding_dim)
            
        with tf.variable_scope("train_test", reuse = True):
            self.decoder_outputs, self.decoder_states_test = tf.contrib.legacy_seq2seq.embedding_rnn_seq2seq(
                self.encoder_inputs, self.decoder_inputs, stacked_lstm, self.xvocab_size, self.yvocab_size, self.embedding_dim, feed_previous=True)


        # For Trainning

        # loss weight: [batch_size, sequence_length]
        loss_weights = [ tf.ones_like(label, dtype=tf.float32) for label in self.labels ]

        # build loss function
        self.loss = tf.contrib.legacy_seq2seq.sequence_loss(self.decoder_outputs, self.labels, loss_weights, self.yvocab_size)

        # train op to minimize the loss
        self.train_op = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.loss)


    '''
        Training and Evaluation

    '''

    # get the feed dictionary
    def get_feed(self, X, Y, keep_prob):
        feed_dict = {self.encoder_inputs[t]: X[t] for t in range(self.xseq_len)}
        feed_dict.update({self.labels[t]: Y[t] for t in range(self.yseq_len)})
        feed_dict[self.keep_prob] = keep_prob # dropout prob
        return feed_dict

    # run one batch for training
    def train_batch(self, sess, train_batch_gen):
        # get batches
        batchX, batchY = train_batch_gen.__next__()
        # build feed
        feed_dict = self.get_feed(batchX, batchY, keep_prob=0.5)
        _, loss_v = sess.run([self.train_op, self.loss], feed_dict)
        return loss_v

    def eval_step(self, sess, eval_batch_gen):
        # get batches
        batchX, batchY = eval_batch_gen.__next__()
        # build feed
        feed_dict = self.get_feed(batchX, batchY, keep_prob=1.)
        loss_v, dec_op_v = sess.run([self.loss, self.decoder_outputs_test], feed_dict)
        # dec_op_v is a list; also need to transpose 0,1 indices 
        #  (interchange batch_size and timesteps dimensions
        dec_op_v = np.array(dec_op_v).transpose([1,0,2])
        return loss_v, dec_op_v, batchX, batchY

    # evaluate 'num_batches' batches
    def eval_batches(self, sess, eval_batch_gen, num_batches):
        losses = []
        for i in range(num_batches):
            loss_v, dec_op_v, batchX, batchY = self.eval_step(sess, eval_batch_gen)
            losses.append(loss_v)
        return np.mean(losses)


    def train(self, train_set, valid_set, sess=None ):
        
        # we need to save the model periodically
        saver = tf.train.Saver()

        # if no session is given
        if not sess:
            # create a session
            sess = tf.Session()
            # init all variables
            sess.run(tf.global_variables_initializer())

        sys.stdout.write('\n<log> Training started </log>\n')
        # run M epochs
        for i in range(self.epochs):
            try:
                self.train_batch(sess, train_set)
                if i and i% (self.epochs//2) == 0: # TODO : make this tunable by the user
                    # save model to disk
                    saver.save(sess, self.ckpt_path + self.model_name + '.ckpt', global_step=i)
                    # evaluate to get validation loss
                    val_loss = self.eval_batches(sess, valid_set, 16) # TODO : and this
                    # print stats
                    print('\nModel saved to disk at iteration #{}'.format(i))
                    print('val   loss : {0:.6f}'.format(val_loss))
                    sys.stdout.flush()
            except KeyboardInterrupt: # this will most definitely happen, so handle it
                print('Interrupted by user at iteration {}'.format(i))
                self.session = sess
                return sess

    def restore_last_session(self):
        saver = tf.train.Saver()
        # create a session
        sess = tf.Session()
        # get checkpoint state
        ckpt = tf.train.get_checkpoint_state(self.ckpt_path)
        # restore session
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
        # return to user
        return sess

    # prediction
    def predict(self, sess, X):
        feed_dict = {self.encoder_inputs[t]: X[t] for t in range(self.xseq_len)}
        feed_dict[self.keep_prob] = 1.
        dec_op_v = sess.run(self.decoder_outputs_test, feed_dict)
        # dec_op_v is a list; also need to transpose 0,1 indices 
        #  (interchange batch_size and timesteps dimensions
        dec_op_v = np.array(dec_op_v).transpose([1,0,2])
        # return the index of item with highest probability
        return np.argmax(dec_op_v, axis=2)
