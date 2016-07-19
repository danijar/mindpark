import numpy as np
import tensorflow as tf


def conv2d(x, filters, size, stride, activation, pooling=None):
    shape = [size, size, int(x.get_shape()[3]), filters]
    weight = tf.Variable(tf.truncated_normal(shape, 0.1), name='conv2d')
    stride = [1, stride, stride, 1]
    x = tf.nn.conv2d(x, weight, stride, 'VALID')
    if pooling:
        pooling = [1, pooling, pooling, 1]
        x = tf.nn.max_pool(x, pooling, pooling, 'VALID')
    return x


def conv3d(x, filters, size, stride, activation, pooling=None):
    shape = [size, size, 1, int(x.get_shape()[4]), filters]
    weight = tf.Variable(tf.truncated_normal(shape, 0.1), name='conv3d')
    stride = [1, stride, stride, 1, 1]
    x = tf.nn.conv3d(x, weight, stride, 'VALID')
    if pooling:
        pooling = [1, pooling, pooling, 1, 1]
        x = tf.nn.max_pool3d(x, pooling, pooling, 'VALID')
    return x


def dense(x, size, activation):
    x = tf.reshape(x, (-1, int(np.prod(x.get_shape()[1:]))))
    x = tf.contrib.layers.fully_connected(x, size, activation)
    return x
