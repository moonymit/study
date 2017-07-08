
# coding: utf-8

# In[1]:

import cv2                 # working with, mainly resizing, images
import numpy as np         # dealing with arrays
import os                  # dealing with directories

import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression

from PIL import Image
import urllib.request
import io

import tensorflow as tf

DIR = os.path.dirname("__file__")

TRAIN_DIR = os.path.join(DIR, './data/train')
TEST_DIR = os.path.join(DIR, './data/test')
IMG_SIZE = 50
LR = 1e-3

MODEL_NAME = 'dogsvscats-{}-{}.model'.format(LR, '2conv-basic')

class CatDogModel:
    def __init__(self):
        tf.reset_default_graph()
        
        convnet = input_data(shape=[None, IMG_SIZE, IMG_SIZE, 1], name='input')

        convnet = conv_2d(convnet, 32, 5, activation='relu')
        convnet = max_pool_2d(convnet, 5)

        convnet = conv_2d(convnet, 64, 5, activation='relu')
        convnet = max_pool_2d(convnet, 5)

        convnet = conv_2d(convnet, 128, 5, activation='relu')
        convnet = max_pool_2d(convnet, 5)

        convnet = conv_2d(convnet, 64, 5, activation='relu')
        convnet = max_pool_2d(convnet, 5)

        convnet = conv_2d(convnet, 32, 5, activation='relu')
        convnet = max_pool_2d(convnet, 5)

        convnet = fully_connected(convnet, 1024, activation='relu')
        convnet = dropout(convnet, 0.8)

        convnet = fully_connected(convnet, 2, activation='softmax')
        convnet = regression(convnet, optimizer='adam', learning_rate=LR, loss='categorical_crossentropy', name='targets')

        self.model = tflearn.DNN(convnet, tensorboard_dir='log')
        
        if os.path.exists(os.path.join(DIR,'./data/{}.meta'.format(MODEL_NAME))):
            self.model.load('./data/{}'.format(MODEL_NAME))
            print('model loaded!')
        else:
            print('model not loaded..')
        
    def train(self, x, y, test_x, test_y, num_epoch = 5):
        self.model.fit({'input': x}, {'targets': y}, n_epoch=num_epoch, validation_set=({'input': test_x}, {'targets': test_y}), 
                  snapshot_step=500, show_metric=True, run_id=MODEL_NAME)
        
        self.model.save('./data/{}'.format(MODEL_NAME))
    
    def predict(self, image_data):
        data = image_data.reshape(IMG_SIZE,IMG_SIZE, 1)
        model_out = self.model.predict([data])[0]
        if np.argmax(model_out) == 1: str_label='Dog'
        else: str_label='Cat'
        return str_label
    


# In[ ]:



