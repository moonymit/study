# Lab 12 Character Sequence RNN
import tensorflow as tf
import numpy as np
tf.set_random_seed(777)  # reproducibility

sample = " if you want you"
idx2char = list(set(sample))
char2idx = {c: i for i, c in enumerate(idx2char)}

# hyper parameters
dic_size = len(char2idx)  # RNN input size (one hot size)
hidden_size = len(char2idx)  # RNN output size
num_classes = len(char2idx)  # final output size (RNN or softmax, etc.)
batch_size = 1  # one sample data, one batch
sequence_length = len(sample) - 1  # number of lstm rollings (unit #)
learning_rate = 0.1

sample_idx = [char2idx[c] for c in sample]
x_data = [sample_idx[:-1]]
y_data = [sample_idx[1:]]

""" Y
[[2 1 3 8 4 0 3 5 7 6 9 3 8 4 0]]
"""
X = tf.placeholder(tf.int32, [None, sequence_length])
Y = tf.placeholder(tf.int32, [None, sequence_length])


""" x_one_hot
[[[ 0.  1.  0.  0.  0.  0.  0.  0.  0.  0.] seq0
  [ 0.  0.  0.  0.  0.  0.  0.  0.  1.  0.] seq1
  [ 0.  0.  0.  0.  0.  0.  0.  0.  0.  1.] seq2
  [ 0.  1.  0.  0.  0.  0.  0.  0.  0.  0.] ...
  [ 0.  0.  0.  0.  0.  1.  0.  0.  0.  0.] ..
  [ 0.  0.  0.  0.  0.  0.  1.  0.  0.  0.] .
  [ 0.  0.  0.  1.  0.  0.  0.  0.  0.  0.]
  [ 0.  1.  0.  0.  0.  0.  0.  0.  0.  0.]
  [ 0.  0.  1.  0.  0.  0.  0.  0.  0.  0.]
  [ 0.  0.  0.  0.  1.  0.  0.  0.  0.  0.]
  [ 1.  0.  0.  0.  0.  0.  0.  0.  0.  0.]
  [ 0.  0.  0.  0.  0.  0.  0.  1.  0.  0.]
  [ 0.  1.  0.  0.  0.  0.  0.  0.  0.  0.]
  [ 0.  0.  0.  0.  0.  1.  0.  0.  0.  0.]
  [ 0.  0.  0.  0.  0.  0.  1.  0.  0.  0.]]]
"""
x_one_hot = tf.one_hot(X, num_classes)

cell = tf.contrib.rnn.BasicLSTMCell(num_units=hidden_size, state_is_tuple=True)
initial_state = cell.zero_state(batch_size, tf.float32)

""" outputs
[[[-0.08187994 -0.04886166 -0.09590879  0.04633991  0.00068239 -0.04437521 -0.0101427   0.077486   -0.09031037 -0.0123993 ]
  [-0.05155265 -0.13920873 -0.11535704 -0.0070303   0.01482051 -0.05271206 -0.04864316  0.01759365 -0.17038503 -0.0690636 ]
  [-0.04830234 -0.14306192 -0.15180127  0.02395121 -0.08279313 -0.03659607  0.01435924 -0.01393827 -0.21907952 -0.01348937]
  [-0.12283339 -0.14884897 -0.19108461  0.06401766 -0.11483106 -0.0655713   0.00094606  0.08366521 -0.28185478 -0.05188923]
  [-0.12446462 -0.14317565 -0.10964032  0.0859701  -0.11853416 -0.10864586  0.08839552  0.05973689 -0.22760159 -0.04443567]
  [-0.16822498 -0.16010138 -0.01308983  0.1207887  -0.16535209 -0.15268138  0.08623657  0.14924733 -0.30124661 -0.03305761]
  [-0.14622021 -0.17490129 -0.01358058  0.07805128 -0.17916366 -0.20872788  0.0587472   0.0911468  -0.2973313  -0.02800302]
  [-0.19664076 -0.19860959 -0.1118517   0.09493651 -0.20148981 -0.19580877  0.01986909  0.170661   -0.32411164 -0.06262434]
  [-0.15699787 -0.22629251 -0.01385089  0.04438514 -0.27731419 -0.19807212  0.0131553   0.19043475 -0.32176116 -0.12973149]
  [-0.1349224  -0.1643053   0.0427383   0.07522171 -0.28965273 -0.0609272  -0.01593936  0.23039506 -0.31669176 -0.07462589]
  [-0.12996727 -0.0868317   0.04487687  0.13913351 -0.29043165 -0.03762217 -0.03411943  0.27231985 -0.35237271 -0.1037432 ]
  [-0.17256679 -0.08034243 -0.00173723  0.12472263 -0.24279712 -0.01681882  0.03934833  0.17110236 -0.27086085 -0.14761348]
  [-0.21233144 -0.11374636 -0.10176529  0.13012531 -0.27817911 -0.0571906   0.01334364  0.22160339 -0.29874331 -0.15960386]
  [-0.19057885 -0.12440286 -0.04802072  0.13068964 -0.21380435 -0.1042501   0.09699173  0.14088525 -0.24758776 -0.12944935]
  [-0.23688647 -0.15253238  0.03079209  0.15082179 -0.24781159 -0.15282789  0.08912186  0.20748036 -0.32185382 -0.08403302]]]
"""

outputs, _states = tf.nn.dynamic_rnn(cell, x_one_hot, initial_state=initial_state, dtype=tf.float32)

weights = tf.ones([batch_size, sequence_length])

"""
logits: A 3D Tensor of shape [batch_size x sequence_length x num_decoder_symbols]
targets: A 2D Tensor of shape [batch_size x sequence_length]
weights: A 2D Tensor of shape [batch_size x sequence_length]
"""
sequence_loss = tf.contrib.seq2seq.sequence_loss(logits=outputs, targets=Y, weights=weights)
loss = tf.reduce_mean(sequence_loss)
train = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(loss)

prediction = tf.argmax(outputs, axis=2)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(3000):
        l, _ = sess.run([loss, train], feed_dict={X: x_data, Y: y_data})
        # result [[7 6 6 6 7 6 7 7 6 6 6 6 6 6 6]]
        result = sess.run(prediction, feed_dict={X: x_data})

        result_str = [idx2char[c] for c in np.squeeze(result)]
        print(i, "loss:", l, "prediction:", ''.join(result_str))

