# coding: utf-8

import numpy as np
import matplotlib.pyplot as plt

def AND(x1, x2):
	x = np.array([x1, x2])
	w = np.array([0.5, 0.5])
	b = -0.7
	tmp = np.sum(w * x) + b

	return step_function(tmp)

def dentity_function(x):
	return x

	
def step_function(x):
	if x > 0:
		return 1
	else
		return 0

def step_function(x):
	y = x > 0
	return y.astype(np.int)

def sigmoid(x):
	return 1 / (1 + np.exp(-x))

def relu(x):
	return np.maxinum(0, x)

