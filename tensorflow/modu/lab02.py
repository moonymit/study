import tensorflow as tf

# train data
x_data = [1, 2, 3]
y_data = [1, 2, 3]

# W should be 1 and b 0
# 초기값을 랜덤으로 준다 -1.0 ~ 1.0
W = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
b = tf.Variable(tf.random_uniform([1], -1.0, 1.0))

X = tf.placeholder(tf.float32);
Y = tf.placeholder(tf.float32);

hypothesis = W * X + b
#hypothesis = W * x_data + b

# reduce_mean : everage
cost = tf.reduce_mean(tf.square(hypothesis - Y)) #operation
#cost = tf.reduce_mean(tf.square(hypothesis - y_data))

# minimize (GradientDescentOptimizer 를 사용해서 cost를 최소화한다 정도로 이해)
a = tf.Variable(0.1) #Learning rate, alpha
optimizer = tf.train.GradientDescentOptimizer(a)
train = optimizer.minimize(cost)
#####################################################################

# initialize Variable(W and b)
init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)


# fit the line
for step in range(2001):
    #sess.run(train) : find W and b
    sess.run(train, feed_dict={X:x_data, Y:y_data})
    if step % 20 == 0:
        print(step, sess.run(cost, feed_dict={X:x_data, Y:y_data}), sess.run(W), sess.run(b))

# We already find W and b
print (sess.run(hypothesis, feed_dict={X:5}))
print (sess.run(hypothesis, feed_dict={X:2.5}))
