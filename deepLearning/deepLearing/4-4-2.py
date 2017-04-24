# -*- coding: utf-8 -*- 

import sys, os
sys.path.append(os.pardir)
import numpy as np
from common.functions import softmax, cross_entropy_error
from common.gradient import numerical_gradient

class simpleNet:
	def __init__(self):
		self.W = np.random.randn(2, 3) #정규분포로 초기화

	def predict(self, x):
		return np.dot(x, self.W)

	def loss(self, x, t):
		z = self.predict(x)
		y = softmax(z)
		loss = cross_entropy_error(y, t)

		return loss

net = simpleNet()
print(net.W)

x = np.array([0.6, 0.9])
p = net.predict(x)
print(p)

result = np.argmax(p) # 최댓값의 인덱스
print(result)

t = np.array([0, 0, 1]) #정답 레이블
result = net.loss(x, t)
print(result)

# def f(W):
# 	return net.loss(x, t)

f = lambda w: net.loss(x, t)

dW = numerical_gradient(f, net.W)
print(dW)

