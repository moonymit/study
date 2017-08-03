
# coding: utf-8

# In[1]:

import gym
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

env = gym.make('CartPole-v0')

# Input and output size based on the Env
input_size = env.observation_space.shape[0] # -> 4
output_size = env.action_space.n # -> 2(left, right)

# Set learning parameters
learning_rate = .1

X = tf.placeholder(tf.float32, [None, input_size], name="input_x") # None will be 1

# First layer of weights
W1 = tf.get_variable("W1", shape=[input_size, output_size], initializer=tf.contrib.layers.xavier_initializer())
Qpred = tf.matmul(X, W1)

# We need to define the parts of the network needed for learning a policy
Y = tf.placeholder(shape=[None, output_size], dtype=tf.float32)

# Loss function
loss = tf.reduce_sum(tf.square(Y - Qpred))

# Learning
train = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

# Set Q-learning related parameters
dis = .99
num_episodes = 2000

# Create lists to contain total rewards and steps per episode
rList = []


# In[3]:

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

sess.run(init)
for i in range(num_episodes):
    # Reset environment and get first new observation
    e = 1. / ((i / 10) + 1)
    step_count = 0
    s = env.reset()
    done = False


    # The Q-Network training
    while not done:
        step_count += 1
        x = np.reshape(s, [1, input_size])

        # Choose an action by freedily (with e change of random action) from the Q-network
        Qs = sess.run(Qpred, feed_dict = {X: x})
        if np.random.rand(1) < e:
            a = env.action_space.sample()
        else:
            a = np.argmax(Qs)

        # Get new state and reward from environment
        s1, reward, done, _ = env.step(a)
        if done:
            # Update Q, and no Qs+1, since it's terminal state
            Qs[0, a] = -100
        else:
            # Obtain the Q_s1 values by feeding the new state through our network
            x1 = np.reshape(s1, [1, input_size])
            Qs1 = sess.run(Qpred, feed_dict={X: x1})
            # Update Q
            Qs[0, a] = reward + dis * np.max(Qs1)

        # Train our network using target (Y) and predicted Q(Qpred) values
        sess.run(train, feed_dict={X:x, Y: Qs})
        s = s1

    rList.append(step_count)
    print("Episode: {} steps: {}".format(i, step_count))

    # If last 10's avg steps are 500, it's good enough
    if len(rList) > 10 and np.mean(rList[-10:]) > 500:
        break;


# In[4]:

# See our trained network in action
observation = env.reset()
reward_sum = 0
while True:
    env.render()
    
    x = np.reshape(observation, [1, input_size])
    Qs = sess.run(Qpred, feed_dict={X:x})
    a = np.argmax(Qs)
    
    observation, reward, done, _ = env.step(a)
    reward_sum += reward
    if done:
        print("Total score: {}".format(reward_sum))
        break


# In[ ]:



