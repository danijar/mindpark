import functools
import numpy as np
import tensorflow as tf


class Model:

    def __init__(self):
        self._placeholders = {}
        self._actions = {}
        self._graph = tf.Graph()
        self._scope = None
        self._cost = None
        self._optimize = None
        self._sess = None

    @property
    def scope(self):
        return self._graph.as_default()

    def __enter__(self):
        self._scope = self._graph.as_default()
        self._scope.__enter__()
        return self

    def __exit__(self, type_, value, traceback):
        self._scope.__exit__(type_, value, traceback)

    def placeholder(self, name, shape=None, type_=tf.float32):
        self._ensure_scope()
        if hasattr(self, name) or name in ('batch_size', 'epochs'):
            raise KeyError('invalid placeholder name ' + name)
        if shape is None:
            shape = (None,)
        elif not hasattr(shape, '__len__'):
            shape = (None, shape)
        else:
            shape = (None,) + shape
        placeholder = tf.placeholder(type_, shape)
        self._placeholders[name] = placeholder
        setattr(self, name, placeholder)

    def action(self, name, action):
        self._ensure_scope()
        if hasattr(self, name):
            raise KeyError('invalid action name ' + name)
        self._actions[name] = action
        setattr(self, name, functools.partial(self._predict, action))

    def compile(self, cost, optimizer=None):
        self._ensure_scope()
        self._cost = tf.reduce_sum(cost)
        optimizer = optimizer or tf.train.RMSPropOptimizer(0.01)
        self._optimize = optimizer.minimize(self._cost)
        self._sess = tf.Session()
        self._sess.run(tf.initialize_all_variables())

    @property
    def variables(self):
        vars_ = self._graph.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        values = self._sess.run(vars_)
        return values

    @variables.setter
    def variables(self, values):
        vars_ = self._graph.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        if len(values) != len(vars_):
            raise ValueError('amount of variables does not fit this model')
        assignments = [var.assign(val) for var, val in zip(vars_, values)]
        self._sess.run(assignments)

    def train(self, batch_size=None, epochs=1, **data):
        for key in data:
            if not key in self._placeholders:
                raise KeyError("invalid placeholder '{}'".format(key))
        keys = sorted(data.keys())
        data = [data[x] for x in keys]
        placeholders = [self._placeholders[x] for x in keys]
        batches = [self._chunks(x, batch_size) for x in data]
        costs = []
        for _ in range(epochs):
            for batch in zip(*batches):
                feed = {k: v for k, v in zip(placeholders, batch)}
                cost, _ = self._sess.run((self._cost, self._optimize), feed)
                costs.append(cost)
        return costs

    def _predict(self, output, **data):
        single = all(
            len(value.shape) + 1 == len(self._placeholders[key].get_shape())
            for key, value in data.items())
        if single:
            data = {k: np.expand_dims(v, 0) for k, v in data.items()}
        feed = {self._placeholders[k]: v for k, v in data.items()}
        result = self._sess.run(output, feed)
        if single:
            result = np.squeeze(result, 0)
        return result

    def _chunks(self, data, size=None):
        if not size:
            yield data
            return
        for index in range(0, len(data), size):
            yield data[index: index + size]

    def _ensure_scope(self):
        if not self._scope:
            raise RuntimeError("use 'with' to build the model")
