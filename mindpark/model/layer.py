import numpy as np
import tensorflow as tf


def conv2d(x, filters, size, stride, activation=tf.nn.relu, pool=None):
    x = tf.contrib.layers.convolution2d(
        x, filters, [size, size], stride, 'VALID',
        activation_fn=activation)
    if pool:
        pool = [1, pool, pool, 1]
        x = tf.nn.max_pool(x, pool, pool, 'VALID')
    return x


def dense(x, size, activation=tf.nn.relu):
    x = tf.reshape(x, (-1, int(np.prod(x.get_shape()[1:]))))
    x = tf.contrib.layers.fully_connected(x, size, activation)
    return x


def rnn(model, x, size, cell=tf.contrib.rnn.GRUCell, activation=tf.tanh):
    x = tf.reshape(x, (-1, int(np.prod(x.get_shape()[1:]))))
    cell = cell(size, activation=activation)
    state = model.add_option('context', cell.zero_state(1, tf.float32))
    x = tf.expand_dims(x, 0)
    x, new_state = tf.nn.dynamic_rnn(cell, x, initial_state=state)
    with tf.control_dependencies([state.assign(new_state)]):
        x = tf.squeeze(x, [0])
    return x
