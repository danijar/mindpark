import numpy as np
import tensorflow as tf


def conv2d(x, filters, size, stride, activation=tf.nn.elu, pool=None):
    shape = [size, size, int(x.get_shape()[3]), filters]
    weight = tf.Variable(tf.truncated_normal(shape, 0.1), name='conv2d')
    stride = [1, stride, stride, 1]
    x = tf.nn.conv2d(x, weight, stride, 'VALID')
    if pool:
        pool = [1, pool, pool, 1]
        x = tf.nn.max_pool(x, pool, pool, 'VALID')
    return x


def conv3d(x, filters, size, stride, activation=tf.nn.elu, pool=None):
    shape = [size, size, 1, int(x.get_shape()[4]), filters]
    weight = tf.Variable(tf.truncated_normal(shape, 0.1), name='conv3d')
    stride = [1, stride, stride, 1, 1]
    x = tf.nn.conv3d(x, weight, stride, 'VALID')
    if pool:
        pool = [1, pool, pool, 1, 1]
        x = tf.nn.max_pool3d(x, pool, pool, 'VALID')
    return x


def rnn(x, size, activation=tf.nn.elu):
    # TODO: Save activation between runs.
    cell = tf.nn.rnn_cell.GRUCell(size, None, activation)
    x, _ = tf.nn.dynamic_rnn(cell, x)
    return x


def dense(x, size, activation=tf.nn.elu):
    x = tf.reshape(x, (-1, int(np.prod(x.get_shape()[1:]))))
    x = tf.contrib.layers.fully_connected(x, size, activation)
    return x


def network_dqn(x, out_size):
    x = conv2d(x, 16, 8, 4, tf.nn.relu)
    x = conv2d(x, 32, 4, 2, tf.nn.relu)
    x = dense(x, 256, tf.nn.relu)
    x = dense(x, out_size, tf.identity)
    return x


def network_my_conv(x, out_size):
    x = conv2d(x, 16, 4, 2, tf.nn.elu, pool=2)
    x = conv2d(x, 32, 3, 1, tf.nn.elu, pool=2)
    x = conv2d(x, 64, 2, 1, tf.nn.elu)
    x = dense(x, 256, tf.nn.elu)
    x = dense(x, 256, tf.nn.elu)
    x = dense(x, out_size, tf.identity)
    return x
