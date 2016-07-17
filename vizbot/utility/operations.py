import numpy as np
import tensorflow as tf


def conv2d(x, filters, size, stride, activation):
    x = tf.contrib.layers.convolution2d(
        x, filters, size, stride, 'VALID', activation)
    return x


def dense(x, size, activation):
    x = tf.reshape(x, (-1, int(np.prod(x.get_shape()[1:]))))
    x = tf.contrib.layers.fully_connected(x, size, activation)
    return x
