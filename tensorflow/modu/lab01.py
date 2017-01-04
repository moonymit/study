import tensorflow as tf

# 1.hello tensorflow
hello = tf.constant("Hellow, TensorFlow!")
sess = tf.Session()

print(sess.run(hello))


# 2.operation
a = tf.constant(2)
b = tf.constant(3)

c = a + b #결과물도 operation

print (c)
print (sess.run(c))


# 3.placeholder
a = tf.placeholder(tf.int16);
b = tf.placeholder(tf.int16);

add = tf.add(a, b);
mul = tf.mul(a, b);

with tf.Session() as sess:
    print ("Addition with variables: %i" % sess.run(add, feed_dict={a: 2, b: 3}))
    print ("Multiplication with variables: %i" % sess.run(mul, feed_dict={a: 2, b: 3}))
