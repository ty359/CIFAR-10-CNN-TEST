import os
import sys

import init

import numpy as np
import tensorflow as tf
import init
import model

sess = tf.InteractiveSession()

_x = tf.placeholder(tf.float32, (model.BATCH_SIZE, 32, 32, 3))
_y = tf.placeholder(tf.float32, (model.BATCH_SIZE, 10))

o = model.build(_x, _y)
model.init(sess)
lo = model.loss()
train_step = tf.train.GradientDescentOptimizer(0.00001).minimize(lo)

correct_prediction = tf.equal(tf.argmax(o,1), tf.argmax(_y,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

for i in range(0, 30000):
	x, y = init.genxy(model.BATCH_SIZE)
	feed_dict = {_x: x, _y: y}
	train_step.run(feed_dict=feed_dict)
	if i % 1000 == 0:
		print(accuracy.eval(feed_dict=feed_dict))