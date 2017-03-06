import os
import sys

import init

import numpy as np
import tensorflow as tf

class NN:

  layer_i = tf.placeholder(tf.float32)
  layer_e = tf.placeholder(tf.float32)
  layers = {}
  loss = 0

  def _fc_layer(x, o_size):
    x = tf.reshape(x, [x.size() / x.shape()[-1] ,x.shape()[-1]])
    k = tf.Variable(tf.truncated_normal([x.shape()[-1], o_size]))
    b = tf.Variable(tf.truncated_normal([o.size]))
    return tf.nn.relu(tf.matmul(x, k) + b)

  def _maxpool_layer(x):
    return tf.nn.max_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

  def _conv_layer(x, o_size):
    k = tf.Variable(tf.truncated_normal([3, 3, x.shape()[3], o_size]))
    b = tf.Variable(tf.truncated_normal([x.shape()[0], x.shape()[1], x.shape()[2], o_size]))
    return tf.nn.relu(tf.nn.conv2d(x, k, [1, 1, 1 ,1], 'SAME') + b)

  def reset_loss():
    loss = 0

  def add_loss(o, e, c):
    loss = loss + c * tf.reduce_sum(tf.map_fn(tf.square, o - e))
  