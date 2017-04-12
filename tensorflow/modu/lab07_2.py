from tensorflow.examples.tutorials.mnist import input_data
from random import randint
import tensorflow as tf
import matplotlib.pyplot as plt

mnist = input_data.read_data_sets('/tmp/tensorflow/mnist/input_data', one_hot=True)

# Create the model
x = tf.placeholder(tf.float32, [None, 784])
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))
k = tf.matmul(x, W) + b
y = tf.nn.softmax(k)

# Define loss and optimizer
y_ = tf.placeholder(tf.float32, [None, 10])
learning_rate = 0.5
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(k, y_))
train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

print ("Training")
sess = tf.Session()
init = tf.global_variables_initializer() #.run()
sess.run(init)
for _ in range(1000):
    #  number of 100 (1000times)
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

print ('b is ',sess.run(b))
print('W is',sess.run(W))

# Get one and predict
r = randint(0, mnist.test.num_examples - 1)
print ("Label: ", sess.run(tf.argmax(mnist.test.labels[r:r+1], 1)))
print ("Prediction: ", sess.run(tf.argmax(y, 1), {x: mnist.test.images[r:r+1]}))

# Show the images
plt.imshow(mnist.test.images[r:r+1].reshape(28, 28), cmap="Greys", interpolation="nearest")
plt.show()
