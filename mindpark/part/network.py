import tensorflow as tf
from mindpark.model.layer import dense, conv2d, rnn


def dqn_2013(model, x):
    # Mnih et al. (2013)
    x = conv2d(x, 16, 8, 4, tf.nn.relu)
    x = conv2d(x, 32, 4, 2, tf.nn.relu)
    x = dense(x, 256, tf.nn.relu)
    return x


def dqn_2015(model, x):
    # Mnih et al. (2015)
    x = conv2d(x, 32, 8, 4, tf.nn.relu)
    x = conv2d(x, 64, 4, 2, tf.nn.relu)
    x = conv2d(x, 64, 3, 1, tf.nn.relu)
    x = dense(x, 512, tf.nn.relu)
    return x


def doom_large(model, x):
    # Kempka et al. (2016)
    x = conv2d(x, 32, 7, 1, tf.nn.relu, 2)
    x = conv2d(x, 32, 5, 1, tf.nn.relu, 2)
    x = conv2d(x, 32, 3, 1, tf.nn.relu, 2)
    x = dense(x, 1024, tf.nn.relu)
    return x


def minecraft_small(model, x):
    # Barron, Whitehead, Yeung (2016)
    x = conv2d(x, 32, 8, 4, tf.nn.relu)
    x = conv2d(x, 64, 4, 2, tf.nn.relu)
    x = dense(x, 512, tf.nn.relu)
    return x


def minecraft_large(model, x):
    # Barron, Whitehead, Yeung (2016)
    x = conv2d(x,  64, 3, 1, tf.nn.relu, 2)
    x = conv2d(x, 128, 3, 1, tf.nn.relu, 2)
    x = conv2d(x, 256, 3, 1, tf.nn.relu, 2)
    x = conv2d(x, 215, 3, 1, tf.nn.relu, 2)
    x = conv2d(x, 215, 3, 1, tf.nn.relu, 2)
    x = dense(x, 4096, tf.nn.relu)
    x = dense(x, 4096, tf.nn.relu)
    return x


def drqn(model, x):
    # Hausknecht, Stone (2015)
    x = conv2d(x, 32, 8, 4, tf.nn.relu)
    x = conv2d(x, 64, 4, 2, tf.nn.relu)
    x = conv2d(x, 64, 3, 1, tf.nn.relu)
    # x = rnn(model, x, 512, tf.nn.rnn_cell.LSTMCell)
    x = rnn(model, x, 512)
    return x


def a3c_lstm(model, x):
    # Mnih et al. (2016)
    x = conv2d(x, 16, 8, 4, tf.nn.relu)
    x = conv2d(x, 32, 4, 2, tf.nn.relu)
    x = dense(x, 256, tf.nn.relu)
    # x = rnn(model, x, 256, tf.nn.rnn_cell.LSTMCell)
    x = rnn(model, x, 256)
    return x


def test(model, x):
    x = dense(x, 8, tf.nn.relu)
    x = dense(x, 8, tf.nn.relu)
    return x


def control(model, x):
    # x = dense(x, 100, tf.nn.relu)
    # x = dense(x, 50, tf.nn.relu)
    x = dense(x, 32, tf.nn.relu)
    x = dense(x, 32, tf.nn.relu)
    return x
