
# In[1]:

import tensorflow as tf
import numpy as np

# preprocessed data
from data import data
import utils

# load data from pickle and npy files
metadata, idx_q, idx_a = data.load_data(PATH='data/')
(trainX, trainY), (testX, testY), (validX, validY) = utils.split_dataset(idx_q, idx_a)

# parameters 
xseq_len = trainX.shape[-1]
yseq_len = trainY.shape[-1]
batch_size = 32
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
                               epochs = 5
                            )

val_batch_gen = utils.rand_batch_gen(validX, validY, batch_size)
train_batch_gen = utils.rand_batch_gen(trainX, trainY, batch_size)

sess = model.restore_last_session()
sess = model.train(train_batch_gen, val_batch_gen, batch_size)
