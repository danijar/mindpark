import numpy as np
import tensorflow as tf
from tensorflow.contrib import layers


def conv2d(x, filters, size, stride, activation=tf.nn.elu, pool=None):
    x = layers.convolution2d(
        x, filters, [size, size], stride, 'VALID', activation)
    if pool:
        pool = [1, pool, pool, 1]
        x = tf.nn.max_pool(x, pool, pool, 'VALID')
    return x


def dense(x, size, activation=tf.nn.elu):
    x = tf.reshape(x, (-1, int(np.prod(x.get_shape()[1:]))))
    x = layers.fully_connected(x, size, activation)
    return x


def rnn(x, size, activation=tf.nn.elu):
    # TODO: Save activation between runs.
    cell = tf.nn.rnn_cell.GRUCell(size, None, activation)
    x, _ = tf.nn.dynamic_rnn(cell, x)
    return x


def network_dqn(x, out_size):
    x = conv2d(x, 16, 8, 4, tf.nn.relu)
    x = conv2d(x, 32, 4, 2, tf.nn.relu)
    x = dense(x, 256, tf.nn.relu)
    x = dense(x, out_size, tf.identity)
    return x


def network_my_1(x, out_size):
    activation = tf.nn.relu
    x = conv2d(x, 16, 4, 2, activation, pool=2)
    x = conv2d(x, 32, 3, 1, activation, pool=2)
    x = conv2d(x, 64, 2, 1, activation)
    x = dense(x, 256, activation)
    x = dense(x, 256, activation)
    x = dense(x, out_size, tf.identity)
    return x


def network_my_2(x, out_size):
    activation = tf.nn.relu
    x = conv2d(x, 16, 8, 1, activation, pool=2)
    x = conv2d(x, 32, 4, 1, activation, pool=2)
    x = conv2d(x, 64, 2, 1, activation)
    x = dense(x, 512, activation)
    x = dense(x, out_size, tf.identity)
    return x


default_network = network_my_2
