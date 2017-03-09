import os
import sys

import init

import numpy as np
import tensorflow as tf

sess = tf.InteractiveSession()

x = tf.Variable([0])

y = tf.Variable([0])

o = (tf.matmul(x, y) - 3) * (tf.matmul(x, y) + 5)

sess.run(tf.global_variables_initializer())

train_step = tf.train.GradientDescentOptimizer(0.5).minimize(o)

for _ in range(0, 100):
	train_step.run()
	print(o.eval())