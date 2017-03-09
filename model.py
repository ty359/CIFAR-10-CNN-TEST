import os
import sys

import init

import numpy as np
import tensorflow as tf

BATCH_SIZE = 100
LEARNING_RATE = 100

losses = []

def get_variable(name, shape, stddev, decay):
  ret = tf.get_variable(name=name, shape=shape, initializer=tf.truncated_normal_initializer(dtype=tf.float32, mean=0.0, stddev=1.0))
  if decay != .0:
    add_loss(ret, tf.zeros(shape), decay)
  return ret

def fc_layer(x, o_size, name):
  with tf.variable_scope(name):
    shape = x.get_shape().as_list()
    batch = shape[0]
    chans = 1
    for i in range(1, len(shape)):
      chans *= shape[i]
    x = tf.reshape(x, [batch, chans])
    k = get_variable('weights', [chans, o_size], 1e-4, 0)
    b = get_variable('biases', [o_size], 1e-4, 0)
    return tf.matmul(x, k) + b

def maxpool_layer(x, name):
  with tf.variable_scope(name):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

def conv_layer(x, o_size, name):
  with tf.variable_scope(name):
    shape = x.get_shape().as_list()
    k = get_variable('weights', [3, 3, shape[3], o_size], 1e-4, 0)
    b = get_variable('biases', [shape[0], shape[1], shape[2], o_size], 1e-4, 0)
    return tf.nn.relu(tf.nn.conv2d(x, k, [1, 1, 1 ,1], 'SAME') + b)

def reset_loss():
  losses = []

def add_loss(o, e, c):
  losses.append((o, e, c))

def build(layer_i, layer_e):
  with tf.variable_scope('test_nn'):
    
    with tf.variable_scope('hidden_1'):
      conv1_1 = conv_layer(layer_i, 8, 'conv1')
      conv1_2 = conv_layer(conv1_1, 8, 'conv2')
      pool1 = maxpool_layer(conv1_2, 'pool')

    with tf.variable_scope('hidden_2'):
      conv2_1 = conv_layer(pool1, 20, 'conv1')
      conv2_2 = conv_layer(conv2_1, 20, 'conv2')
      pool2 = maxpool_layer(conv2_2, 'pool')

    with tf.variable_scope('hidden_3'):
      conv3_1 = conv_layer(pool2, 20, 'conv1')
      conv3_2 = conv_layer(conv2_1, 20, 'conv2')
      pool3 = maxpool_layer(conv2_2, 'pool')
    
    with tf.variable_scope('summize'):
      out = fc_layer(layer_i, layer_e.get_shape().as_list()[1], 'fc')
      add_loss(out, layer_e, LEARNING_RATE)

    return out

def init(sess):
  sess.run(tf.global_variables_initializer())

def loss():
  ret = 0
  for l in losses:
    ret += l[2] * tf.reduce_sum(tf.nn.l2_loss(l[0] - l[1]))
  return ret