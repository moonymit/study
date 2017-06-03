
import tensorflow as tf
import numpy as np

# preprocessed data
from data import data
import utils

from random import sample

# load data from pickle and npy files
metadata, idx_q, idx_a = data.load_data(PATH='data/')
(trainX, trainY), (testX, testY), (validX, validY) = utils.split_dataset(idx_q, idx_a)

# parameters 
xseq_len = trainX.shape[-1]
yseq_len = trainY.shape[-1]

batch_size = 16
val_batch_size = len(validX)
train_batch_size = len(trainX)

xvocab_size = len(metadata['idx2w'])  
yvocab_size = xvocab_size
emb_dim = 1024

import tars_seq2seq

model = tars_seq2seq.Seq2Seq(xseq_len=xseq_len,
                               yseq_len=yseq_len,
                               xvocab_size=xvocab_size,
                               yvocab_size=yvocab_size,
                               ckpt_path='ckpt/',
                               embedding_dim=emb_dim,
                               num_layers=3,
                               epochs = 500,
                               save_freq = 5
                            )


# print(len(validY), len(validX)) # 7, 7
# print(len(trainY), len(trainX)) # 33, 33

# batch_size must smaller then validX(trainX)
val_batch_gen = utils.rand_batch_gen(validX, validY, val_batch_size)
train_batch_gen = utils.rand_batch_gen(trainX, trainY, train_batch_size)

sess = model.restore_last_session()
# sess = model.train(train_batch_gen, val_batch_gen, val_batch_size)


# TEST
idx_sentence = idx_q[8];
print(idx_sentence)

sentence = "멍청아"
sentence_tokenized = sentence.split(' ')
idx_sentence = data.zero_pad_q(sentence_tokenized, metadata['w2idx'])
print(idx_sentence, sentence_tokenized)

idx_sentence_len = len(idx_sentence)
print(idx_sentence, idx_sentence_len)
print(idx_sentence.reshape(idx_sentence_len, 1).shape)

result = model.predict(sess, idx_sentence.reshape(idx_sentence_len, 1));
idx_words = [ metadata['idx2w'][idx_word] for idx_word in result[0]]
print(idx_words)
