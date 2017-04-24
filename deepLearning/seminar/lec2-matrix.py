# coding: utf-8

import numpy as np

A = np.array([[1, 2],
			  [3, 4]])

B = np.array([[5, 6],
			  [7, 8]])

r = np.dot(A, B)

print(r)

X = np.array([1, 2])

W = np.array([[1, 3, 5],
			  [2, 4, 6]])

Y  = np.dot(X, W)

print(Y)