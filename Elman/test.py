from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

# NN parameters
n_hidden_1 = 1
n_input = 3
n_classes = 1

prototypes = tf.constant([[1,-1,-1],[1,1,-1]], tf.float32)
prototypes_class = tf.constant([0,1], tf.float32)

X = tf.placeholder("float", [None, n_input,n_hidden_1])

weights = {
	#'h1': tf.Variable(tf.random_normal([n_hidden_1, n_input]))
	'h1' : tf.Variable([0.5,-1,-0.5])
}

biases = {
	#'b1': tf.Variable(tf.random_normal([n_hidden_1]))
	'b1' :tf.Variable([0.5])
}

def hardlim (a):
	if a <= 0:
		value_h = 0
	else:
		value_h = 1
	return value_h

def flip (b):
	if b <= 0:
		value_f = 1
	else:
		value_f = 0
	return value_f

def neural_net (x):
	layer_1 = tf.reduce_sum(x * weights['h1'])  + biases['b1']
	return layer_1

def train_w (e, w, p):
	w = (e * p) + w
	return w

def train_b (e, b):
	b = e + b
	return b

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)
i = 0
logits = neural_net (prototypes[i, :])
error = prototypes_class[i] - hardlim(sess.run(logits))

print("Start")

while error != "0.0":
	#sess.run(train_w(error,weights['h1'],prototypes[i]))
	logits = neural_net (prototypes[i, :])
	error = prototypes_class[i] - hardlim(sess.run(logits))
	weights['h1'] = train_w(error,weights['h1'],prototypes[i, :])
	biases['b1'] = train_b(error,biases['b1'])

	print("weights")
	print(sess.run(weights['h1']))
	print("error")
	print(sess.run(error))
	print('bias')
	print(sess.run(biases['b1']))
	print("logits")
	print(sess.run(logits))
	print("Next iteration")
	i = flip(i)

print ("End")