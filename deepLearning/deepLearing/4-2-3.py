# coding: utf-8


import sys, os
sys.path.append(os.pardir)
import numpy as np
from dataset.mnist import load_mnist

(x_train, t_train), (x_test, t_test) = load_mnist(normalize = True, one_hot_label=True)

print(x_train.shape)
print(t_train.shape)

train_size = x_train.shape[0]
batch_size = 10
batch_mask = np.random.choice(train_size, batch_size) #0~trin_size 중에 batch_size 만큼 무작위로 골라냄
x_batch = x_train[batch_mask]
t_batch = t_train[batch_mask]

print(batch_mask);

print(t_batch)
print(t_batch.reshape(1, t_batch.size))
print(t_batch.ndim);
print(t_batch.size);

# one_hot_encoding = true
def cross_entropy_error(y, t):
	if y.ndim == 1: # y가 1차원이라면 : 데이터가 한개일 때
		t = t.reshape(1, t.size)
		y = y.reshape(1, y.size)

	batch_size = y.shape[0]
	return -np.sum(t * np.log(y)) / batch_size

# one_hot_encoding = false
def cross_entropy_error(y, t):
	if y.ndim == 1: # y가 1차원이라면 데이터가 한개일 떄 
		t = t.reshape(1, t.size)
		y = y.reshape(1, y.size)

	batch_size = y.shape[0]
	return -np.sum(np.log(y[np.arange(batch_size), t])) / batch_size

