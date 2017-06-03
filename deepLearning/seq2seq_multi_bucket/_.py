import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

import random

import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

import data_utils

w_t = tf.get_variable("proj_w", [10, 5], dtype=tf.float32)
print(w_t)


W = tf.Variable(tf.random_normal([10]), name='weight')
print(W)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

print(sess.run(w_t))
print(sess.run(W))


x = tf.placeholder(tf.float32, (None,), 'x_1')
y = tf.reduce_sum(x)

y2 = x

print(x.name)
print(sess.run([y, y2], feed_dict = {x.name: [1, 2, 3]}))